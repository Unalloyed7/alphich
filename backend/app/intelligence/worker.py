"""Separate LLM worker: leases, heartbeats, no automatic retry of paid calls."""
from __future__ import annotations

import argparse
import logging
import time

from sqlalchemy import or_, select, update

from app.core import Base, make_session_factory
from app.news import Job
from .service import Runtime
from .settings import AnalysisError, Settings
from .storage import AnalysisTask

LOG = logging.getLogger(__name__)
LEASE_SECONDS = 600


def claim(factory, now=None):
    now = int(time.time()) if now is None else now
    operations = ("insight_analysis", "insight_question")
    with factory.begin() as db:
        db.execute(update(Job).where(Job.operation.in_(operations), Job.status == "running",
            Job.attempt_count >= 3, Job.lease_expires_at <= now).values(
                status="failed", stage="failed", error="analysis_lease_expired", lease_expires_at=None, finished_at=now))
        eligibility = or_(Job.status == "queued", (Job.status == "running") & (Job.lease_expires_at <= now))
        for job in db.scalars(select(Job).where(Job.operation.in_(operations), eligibility,
                Job.available_at <= now, Job.attempt_count < 3).order_by(Job.created_at, Job.id).limit(20)).all():
            attempt = job.attempt_count + 1
            changed = db.execute(update(Job).where(Job.id == job.id, Job.attempt_count == job.attempt_count,
                eligibility).values(status="running", stage="llm_analysis", attempt_count=attempt,
                                    heartbeat_at=now, lease_expires_at=now + LEASE_SECONDS, error=None))
            if changed.rowcount == 1:
                return job.id, attempt
    return None


def process_next(factory, runtime, now=None):
    acquired = claim(factory, now)
    if acquired is None:
        return False
    job_id, attempt = acquired

    def heartbeat():
        at = int(time.time())
        with factory.begin() as db:
            changed = db.execute(update(Job).where(Job.id == job_id, Job.status == "running",
                Job.attempt_count == attempt, Job.lease_expires_at > at).values(
                    heartbeat_at=at, lease_expires_at=at + LEASE_SECONDS))
            if changed.rowcount != 1:
                raise AnalysisError("analysis_lease_lost")
    try:
        with factory() as db:
            job, task = db.get(Job, job_id), db.get(AnalysisTask, job_id)
            if task is None:
                raise AnalysisError("analysis_task_missing")
            db.expunge(job)
            db.expunge(task)
        result_type, result_id = runtime.process(task, job, heartbeat)
        heartbeat()
        with factory.begin() as db:
            db.execute(update(Job).where(Job.id == job_id, Job.status == "running", Job.attempt_count == attempt).values(
                status="succeeded", stage="analysis_verified", result_type=result_type, result_id=result_id,
                lease_expires_at=None, heartbeat_at=int(time.time()), finished_at=int(time.time())))
    except Exception as exc:
        code = exc.code if isinstance(exc, AnalysisError) else "analysis_failed"
        # Do not log exception repr: upstream transport errors can contain secrets/source text.
        LOG.warning("Analysis job %s failed: %s", job_id, code)
        with factory.begin() as db:
            values = dict(status="failed", stage="budget_exceeded" if code == "budget_exceeded" else "analysis_failed",
                          error=code, lease_expires_at=None, finished_at=int(time.time()))
            if code in {"extraction_in_progress", "analysis_in_progress"}:
                values.update(status="queued", stage="waiting_shared_analysis", available_at=int(time.time()) + 5,
                              attempt_count=attempt - 1, finished_at=None)
            db.execute(update(Job).where(Job.id == job_id, Job.status == "running", Job.attempt_count == attempt).values(**values))
    return True


def main():
    parser = argparse.ArgumentParser(description="Bounded Alphich LLM worker")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    engine, factory = make_session_factory()
    Base.metadata.create_all(engine)
    runtime = Runtime(factory, Settings.from_env())
    try:
        while True:
            processed = process_next(factory, runtime)
            if args.once:
                break
            if not processed:
                time.sleep(1)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
