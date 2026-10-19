"""Single-transaction news ingestion worker; no LLM calls or fake fact extraction.

Safe for multiple PostgreSQL workers through conditional status transitions.
A claimed job is leased and can be reclaimed after a worker crash. A stale
worker cannot publish results if its lease is no longer the active claim.
"""
from __future__ import annotations

import argparse
import logging
import time

from sqlalchemy import or_, select, update
from sqlalchemy.orm import sessionmaker

from app.core import Base, make_session_factory
from app.news import Event, Job, SourceDocument, new_id

LOG = logging.getLogger(__name__)
MAX_ATTEMPTS = 3
LEASE_SECONDS = 90


def claim_job(session_factory: sessionmaker, *, now: int | None = None) -> tuple[str, int] | None:
    """Compare-and-swap claim; returns job ID plus unique attempt generation."""
    now = int(time.time()) if now is None else now
    with session_factory.begin() as db:
        # Third crashed attempt cannot be reclaimed; do not leave it running forever.
        db.execute(update(Job).where(
            Job.status == "running", Job.attempt_count >= MAX_ATTEMPTS,
            Job.lease_expires_at <= now,
        ).values(status="failed", stage="failed", error="worker_lease_expired",
                 finished_at=now, lease_expires_at=None))
        candidates = db.scalars(
            select(Job).where(
                Job.available_at <= now,
                Job.attempt_count < MAX_ATTEMPTS,
                or_(Job.status == "queued", (Job.status == "running") & (Job.lease_expires_at <= now)),
            ).order_by(Job.created_at, Job.id).limit(20)
        ).all()
        for candidate in candidates:
            attempt = candidate.attempt_count + 1
            eligible = or_(Job.status == "queued", (Job.status == "running") & (Job.lease_expires_at <= now))
            changed = db.execute(
                update(Job).where(
                    Job.id == candidate.id,
                    Job.attempt_count == candidate.attempt_count,
                    Job.available_at <= now,
                    eligible,
                ).values(status="running", stage="creating_event", attempt_count=attempt,
                         lease_expires_at=now + LEASE_SECONDS, heartbeat_at=now, error=None)
            )
            if changed.rowcount == 1:
                return candidate.id, attempt
    return None


def process_claim(session_factory: sessionmaker, job_id: str, attempt: int) -> bool:
    """Create at most one event per source and atomically mark the job complete.

    If a lease was stolen, do not publish a stale result. Concurrency with
    multiple different jobs for one source is serialized by source row locking
    in PostgreSQL (and SQLite serializes writes).
    """
    now = int(time.time())
    with session_factory.begin() as db:
        job = db.scalar(select(Job).where(Job.id == job_id).with_for_update())
        if job is None or job.status != "running" or job.attempt_count != attempt or (job.lease_expires_at or 0) <= now:
            return False
        # Locking source serializes two jobs for the same exact publication in PostgreSQL.
        source = db.scalar(select(SourceDocument).where(SourceDocument.id == job.source_id).with_for_update())
        if source is None:
            raise ValueError("source_not_found")
        event = db.scalar(select(Event).where(Event.source_id == source.id))
        if event is None:
            event = Event(
                id=new_id("event"), company_id=source.company_id, source_id=source.id,
                title=source.title, published_at=source.published_at,
                event_at=None, topics=[], topic_method="not_classified",
                extraction_status="pending_extraction", completeness=source.completeness,
                version=1,
            )
            db.add(event)
            db.flush()
        job.status = "succeeded"
        job.stage = "stored_raw_event"
        job.result_id = event.id
        job.result_type = "event"
        job.error = None
        job.lease_expires_at = None
        job.heartbeat_at = now
        job.finished_at = now
        source.extraction_status = "raw_only"
        return True


def fail_claim(session_factory: sessionmaker, job_id: str, attempt: int) -> None:
    now = int(time.time())
    with session_factory.begin() as db:
        job = db.get(Job, job_id)
        if job is None or job.status != "running" or job.attempt_count != attempt:
            return
        if attempt >= MAX_ATTEMPTS:
            job.status = "failed"
            job.stage = "failed"
            job.finished_at = now
        else:
            job.status = "queued"
            job.stage = "retry_waiting"
            job.available_at = now + 2 ** attempt
        job.lease_expires_at = None
        job.heartbeat_at = now
        job.error = "ingestion_failed"  # Never expose source text or internal exceptions.


def process_next_job(session_factory: sessionmaker, *, now: int | None = None) -> bool:
    claimed = claim_job(session_factory, now=now)
    if claimed is None:
        return False
    job_id, attempt = claimed
    try:
        process_claim(session_factory, job_id, attempt)
    except Exception:
        LOG.exception("Publication job failed, id=%s attempt=%s", job_id, attempt)
        fail_claim(session_factory, job_id, attempt)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Alphich publication ingestion worker")
    parser.add_argument("--once", action="store_true", help="Attempt processing one job and exit")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    args = parser.parse_args()
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    logging.basicConfig(level=logging.INFO)
    engine, factory = make_session_factory()
    Base.metadata.create_all(engine)  # MVP bootstrap; replace with migrations.
    if args.once:
        process_next_job(factory)
        return
    LOG.info("Ingestion worker started")
    try:
        while True:
            if not process_next_job(factory):
                time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        LOG.info("Worker stopped")
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
