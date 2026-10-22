"""Opt-in, bounded RSS -> shared facts -> grades -> selected personal cards.

The collector never calls models. This dispatcher creates durable work for the
analysis worker; low grades never create a personal generation task.
"""
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import os
import time

from sqlalchemy import JSON, Integer, String, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from app.core import Base, Company
from app.news import Event, Job, SourceDocument, new_id
from .article_cache import _configuration
from .provider import hashed
from .settings import AnalysisError
from .storage import AnalysisTask, Extraction

AUTOMATION_VERSION = "rss-business-pipeline-v1"


class AutomationRecord(Base):
    __tablename__ = "news_automation_decisions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    role: Mapped[str] = mapped_column(String(8))
    event_id: Mapped[str] = mapped_column(String(40), index=True)
    source_id: Mapped[str] = mapped_column(String(40))
    profile_hash: Mapped[str] = mapped_column(String(64))
    as_of: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(48))
    grade: Mapped[int | None] = mapped_column(Integer)
    assessment: Mapped[dict] = mapped_column(JSON, default=dict)
    extraction_job_id: Mapped[str | None] = mapped_column(String(40))
    analysis_job_id: Mapped[str | None] = mapped_column(String(40))
    error: Mapped[str | None] = mapped_column(String(80))
    updated_at: Mapped[int] = mapped_column(Integer)


class AutomationQuota(Base):
    __tablename__ = "news_automation_daily_quotas"
    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    extraction_jobs: Mapped[int] = mapped_column(Integer, default=0)
    card_jobs: Mapped[int] = mapped_column(Integer, default=0)


@dataclass(frozen=True)
class AutomationConfig:
    enabled: bool = False
    company_ids: tuple[str, ...] = ()
    roles: tuple[str, ...] = ("cfo",)
    extraction_limit: int = 20
    card_limit: int = 10
    scan_limit: int = 1000
    max_age_days: int = 90

    @classmethod
    def from_env(cls):
        flag = os.getenv("AI_AUTOMATION_ENABLED", "0")
        if flag not in {"0", "1"}:
            raise ValueError("invalid_automation_flag")
        allowed = tuple(sorted({s.strip() for s in os.getenv("DEMO_COMPANY_IDS", "").split(",") if s.strip()}))
        selected = tuple(sorted({s.strip() for s in os.getenv("AI_AUTOMATION_COMPANY_IDS", "").split(",") if s.strip()})) or allowed
        config = cls(flag == "1", selected,
            tuple(s.strip() for s in os.getenv("AI_AUTOMATION_ROLES", "cfo").split(",") if s.strip()),
            int(os.getenv("AI_AUTOMATION_MAX_EXTRACTIONS", "20")),
            int(os.getenv("AI_AUTOMATION_MAX_CARDS", "10")),
            int(os.getenv("AI_AUTOMATION_SCAN_LIMIT", "1000")))
        config.validate(allowed)
        return config

    def validate(self, allowed):
        if (not set(self.company_ids) <= set(allowed) or len(self.company_ids) > 20
                or self.enabled and not self.company_ids or not self.roles
                or len(set(self.roles)) != len(self.roles) or set(self.roles) - {"cfo", "ceo"}
                or not 1 <= self.extraction_limit <= 100 or not 1 <= self.card_limit <= 100
                or not 1 <= self.scan_limit <= 5000 or not 1 <= self.max_age_days <= 90):
            raise ValueError("invalid_automation_scope_or_limits")


def _quota(db, day, field, limit):
    insert = sqlite_insert if db.bind.dialect.name == "sqlite" else pg_insert
    db.execute(insert(AutomationQuota).values(day=day, extraction_jobs=0, card_jobs=0)
        .on_conflict_do_nothing(index_elements=["day"]))
    column = getattr(AutomationQuota, field)
    return db.execute(update(AutomationQuota).where(AutomationQuota.day == day, column < limit)
        .values(**{field: column + 1})).rowcount == 1


def _job(db, *, key, operation, event, source, profile, role, as_of, config, now):
    owner = hashed(["automation", event.company_id])
    previous = db.scalar(select(Job).where(Job.owner_session_hash == owner, Job.operation == operation,
        Job.idempotency_key == key))
    if previous:
        return previous, False
    field = "extraction_jobs" if operation == "news_fact_extraction" else "card_jobs"
    cap = config.extraction_limit if field == "extraction_jobs" else config.card_limit
    if not _quota(db, datetime.fromtimestamp(now, timezone.utc).date().isoformat(), field, cap):
        return None, False
    job = Job(id=new_id("job"), owner_session_hash=owner, company_id=event.company_id, operation=operation,
        idempotency_key=key, request_hash=hashed([key, operation]), source_id=source.id, status="queued",
        stage="waiting_facts" if operation == "news_fact_extraction" else "waiting_selected_analysis",
        result_type="event" if operation == "news_fact_extraction" else "insight", attempt_count=0,
        created_at=now, available_at=now)
    db.add(job)
    db.add(AnalysisTask(job_id=job.id, event_id=event.id, role=role, as_of=as_of, profile=profile))
    db.flush()
    return job, True


def dispatch_once(factory, settings, config=None, *, now=None, event_ids=None):
    from alphich_kb.personalization.grading import grade_event
    from alphich_kb.personalization.grading import VERSION as GRADING_VERSION
    from alphich_kb.personalization import VERSION as SELECTION_VERSION
    from alphich_kb.personalization.inputs import SelectionScope, instant
    from alphich_kb.personalization.profiles import normalize_company
    from alphich_kb.personalization.rules import match_reasons
    from .service import company_profile, normalized_event
    from .news_index import target_ids
    config = config or AutomationConfig.from_env()
    config.validate(target_ids())
    report = {"enabled": config.enabled, "scanned": 0, "extractions_queued": 0, "cards_queued": 0,
              "below_threshold": 0, "awaiting_quota": 0, "errors": []}
    if not config.enabled:
        return report
    try:
        settings.require_ready()
    except (AnalysisError, ValueError):
        report["errors"].append("llm_not_configured")
        return report
    now = int(time.time()) if now is None else now
    when = datetime.fromtimestamp(now, timezone.utc)
    as_of = when.isoformat()
    since = (when - timedelta(days=config.max_age_days)).isoformat()
    with factory() as db:
        ids = list(db.scalars(select(Event.id).join(SourceDocument, Event.source_id == SourceDocument.id).where(
            Event.company_id.in_(config.company_ids), SourceDocument.company_id == Event.company_id,
            SourceDocument.kind == "rss_article", SourceDocument.published_at >= since,
            SourceDocument.published_at <= as_of, SourceDocument.received_at <= as_of)
            .order_by(SourceDocument.received_at.desc(), Event.id).limit(config.scan_limit)))
    configuration = _configuration(settings)
    if event_ids is not None:
        ids = [event_id for event_id in ids if event_id in frozenset(event_ids)]
    for event_id in ids:
        report["scanned"] += 1
        try:
            with factory.begin() as db:
                event = db.get(Event, event_id)
                source = db.get(SourceDocument, event.source_id)
                from app.rss import RssSourcePolicy
                policy = db.get(RssSourcePolicy, source.id)
                if policy is not None and policy.suppress_automatic_ai:
                    continue  # A raw test collection is analyzed only on an explicit AI request.
                # A spoofed source kind is not collector provenance.
                from .article_cache import _public_snapshot
                if _public_snapshot(db, source) is None:
                    continue
                company = db.get(Company, event.company_id)
                if company is None:
                    continue
                extraction_key = "extraction_" + hashed([source.id, source.version, source.content_hash, configuration])
                extracted = db.get(Extraction, extraction_key)
                current = normalized_event(db, event, source, extracted if extracted and extracted.status == "completed" else None)
                for role in config.roles:
                    profile = company_profile(company, role)
                    profile_hash = hashed(profile)
                    key = hashed([AUTOMATION_VERSION, SELECTION_VERSION, GRADING_VERSION, source.id, source.version, source.content_hash,
                                  profile_hash, role, configuration])
                    record = db.get(AutomationRecord, key)
                    if record is None:
                        record = AutomationRecord(id=key, company_id=company.id, role=role, event_id=event.id,
                            source_id=source.id, profile_hash=profile_hash, as_of=as_of, status="pending", updated_at=now)
                        db.add(record)
                    if record.status in {"irrelevant", "below_threshold", "completed", "blocked"}:
                        continue
                    scope = SelectionScope(company.id, record.as_of, frozenset({event.id}), role)
                    if scope.excludes(current):
                        continue
                    normalized_profile = normalize_company(profile)
                    reasons, _ = match_reasons(normalized_profile, current, scope)
                    if not extracted or extracted.status != "completed":
                        if not reasons:
                            record.status, record.grade = "irrelevant", 0
                            record.updated_at = now
                            continue
                        fact_key = hashed(["facts", source.id, source.version, source.content_hash, configuration])
                        job, created = _job(db, key=fact_key, operation="news_fact_extraction", event=event,
                            source=source, profile=profile, role=role, as_of=record.as_of, config=config, now=now)
                        if job is None:
                            record.status = "awaiting_extraction_quota"
                            report["awaiting_quota"] += 1
                            continue
                        record.extraction_job_id = job.id
                        record.status = "blocked" if job.status == "failed" else "waiting_extraction"
                        record.error = job.error
                        report["extractions_queued"] += int(created)
                        continue
                    grading = grade_event(normalized_profile, current, scope, reasons)
                    record.grade, record.assessment, record.updated_at = (
                        grading["relevance_grade"], grading["relevance_assessment"], now)
                    if not grading["relevance_selected"]:
                        record.status = "below_threshold"
                        report["below_threshold"] += 1
                        continue
                    job, created = _job(db, key=key, operation="automatic_insight", event=event, source=source,
                        profile=profile, role=role, as_of=record.as_of, config=config, now=now)
                    if job is None:
                        record.status = "awaiting_card_quota"
                        report["awaiting_quota"] += 1
                        continue
                    record.analysis_job_id = job.id
                    record.status = "completed" if job.status == "succeeded" else "blocked" if job.status == "failed" else "waiting_analysis"
                    record.error = job.error
                    report["cards_queued"] += int(created)
        except IntegrityError:
            # Another dispatcher won the same immutable input; the entire
            # decision/quota/job transaction is rolled back, including its quota.
            report["errors"].append("concurrent_dispatch")
    return report


def main():
    import argparse
    import json
    from app.core import make_session_factory
    from .gateway import load_env_file
    from .settings import Settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["status", "tick"])
    parser.add_argument("--env-file")
    args = parser.parse_args()
    if args.env_file:
        load_env_file(args.env_file)
    config = AutomationConfig.from_env()
    engine, factory = make_session_factory()
    Base.metadata.create_all(engine)
    try:
        if args.command == "tick":
            print(json.dumps(dispatch_once(factory, Settings.from_env(), config), ensure_ascii=False))
        else:
            from sqlalchemy import func
            with factory() as db:
                rows = db.execute(select(AutomationRecord.status, func.count()).where(
                    AutomationRecord.company_id.in_(config.company_ids)).group_by(AutomationRecord.status)).all()
            print(json.dumps({"enabled": config.enabled, "company_ids": config.company_ids,
                "roles": config.roles, "daily_extraction_job_limit": config.extraction_limit,
                "daily_card_job_limit": config.card_limit, "counts": dict(rows)}, ensure_ascii=False))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
