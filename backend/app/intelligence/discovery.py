"""User-triggered durable news discovery. Only the existing worker makes paid calls."""
from datetime import datetime, timedelta, timezone
import time

from sqlalchemy import JSON, Integer, String, UniqueConstraint, and_, or_, select, update
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.orm.attributes import flag_modified

from app.core import Base, Company
from app.news import Event, Job, SourceDocument, new_id
from .article_cache import _configuration, _public_snapshot
from .provider import hashed
from .settings import AnalysisError
from .storage import AnalysisTask, BudgetDay, Extraction, Insight, LLMCall

ACTIVE = ('collecting', 'waiting_ingestion', 'analyzing', 'stopping')
TERMINAL = ('completed', 'partial', 'cancelled', 'failed')
MAX_SECONDS = 1800


class DiscoveryRun(Base):
    __tablename__ = 'news_discovery_runs'
    __table_args__ = (UniqueConstraint('owner_session_hash', 'idempotency_key'),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    role: Mapped[str] = mapped_column(String(8))
    owner_session_hash: Mapped[str] = mapped_column(String(64))
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    active_key: Mapped[str | None] = mapped_column(String(120), unique=True)
    status: Mapped[str] = mapped_column(String(32))
    stage: Mapped[str] = mapped_column(String(60))
    stop_reason: Mapped[str | None] = mapped_column(String(80))
    target: Mapped[int] = mapped_column(Integer)
    min_grade: Mapped[int] = mapped_column(Integer)
    max_candidates: Mapped[int] = mapped_column(Integer)
    profile: Mapped[dict] = mapped_column(JSON)
    items: Mapped[list] = mapped_column(JSON, default=list)
    collection: Mapped[list] = mapped_column(JSON, default=list)
    stats: Mapped[dict] = mapped_column(JSON, default=dict)
    feed_cursor: Mapped[int] = mapped_column(Integer, default=0)
    collection_lease: Mapped[str | None] = mapped_column(String(40))
    collection_lease_until: Mapped[int] = mapped_column(Integer, default=0)
    as_of: Mapped[str | None] = mapped_column(String(40))
    budget_day: Mapped[str] = mapped_column(String(10))
    created_at: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[int] = mapped_column(Integer)
    deadline_at: Mapped[int] = mapped_column(Integer)


def finish(run, status, reason):
    run.status, run.stage, run.stop_reason, run.active_key = status, 'finished', reason, None
    run.updated_at = int(time.time())


def write_items(run, items):
    # enqueue_child flushes the transaction before its returned job ID is added.
    # Explicitly mark nested JSON changes after that flush as well.
    run.items = [dict(item) for item in items]
    flag_modified(run, 'items')


def view_run(db, run):
    items = run.items or []
    job_ids = [j for item in items for j in (item.get('facts_job_id'), item.get('analysis_job_id')) if j]
    run_ids = ['run_' + hashed(j)[:32] for j in job_ids]
    calls = list(db.scalars(select(LLMCall).where(LLMCall.run_id.in_(run_ids)))) if run_ids else []
    counts = {state: sum(i['state'] == state for i in items) for state in
              ('completed', 'below_threshold', 'failed', 'waiting_facts', 'waiting_analysis', 'pending')}
    return {'id': run.id, 'company_id': run.company_id, 'role': run.role, 'status': run.status,
        'stage': run.stage, 'stop_reason': run.stop_reason, 'target': run.target, 'min_grade': run.min_grade,
        'max_candidates': run.max_candidates, 'deadline_at': run.deadline_at, 'created_at': run.created_at,
        'updated_at': run.updated_at, 'stats': {**run.stats, **counts,
            'checked': sum(i.get('grade') is not None for i in items),
            'feeds_checked': run.feed_cursor, 'reused': sum(bool(i.get('reused')) for i in items)},
        'cost_rub': str(sum(c.cost_micros or 0 for c in calls) / 1_000_000),
        'cost_unknown': any(c.cost_micros is None for c in calls),
        'sources': run.collection, 'items': [{k: v for k, v in item.items() if k not in
            {'facts_job_id', 'analysis_job_id', 'source_revision'}} for item in items]}


def collect_tick(factory, *, feeds=None, poll=None, now=None):
    """One feed per free ingestion-worker tick; the API never performs network collection."""
    from app.rss import load_config, poll_feed, target_companies
    now = int(time.time()) if now is None else now
    poll = poll or poll_feed
    with factory.begin() as db:
        run = db.scalar(select(DiscoveryRun).where(DiscoveryRun.status == 'collecting',
            DiscoveryRun.collection_lease_until <= now).order_by(DiscoveryRun.created_at)
            .with_for_update(skip_locked=True).limit(1))
        if run is None:
            return False
        try:
            feeds = load_config() if feeds is None else feeds
        except Exception:
            finish(run, 'failed', 'invalid_rss_config')
            return True
        if run.deadline_at <= now:
            finish(run, 'partial', 'time_limit')
            return True
        try:
            target_companies([run.company_id])
        except Exception:
            finish(run, 'failed', 'company_unavailable')
            return True
        if run.feed_cursor >= len(feeds):
            run.status, run.stage = 'waiting_ingestion', 'waiting_for_articles'
            return True
        token, index = new_id('lease'), run.feed_cursor
        run.collection_lease, run.collection_lease_until = token, now + 900
        run.stage, run.updated_at = 'collecting_sources', now
        run_id, company_id = run.id, run.company_id
    try:
        result = poll(factory, feeds[index], [company_id], force=True)
    except Exception:
        result = {'feed_id': feeds[index].id, 'status': 'failed', 'errors': ['collection_failed']}
    with factory.begin() as db:
        run = db.get(DiscoveryRun, run_id)
        if run.collection_lease == token:
            run.collection = [*run.collection, result]
            run.feed_cursor, run.collection_lease, run.collection_lease_until = index + 1, None, 0
            run.updated_at = int(time.time())
            if run.status == 'collecting' and run.feed_cursor >= len(feeds):
                run.status, run.stage = 'waiting_ingestion', 'waiting_for_articles'
    return True


def configured_article_ids(db):
    """A cached article must still belong to a currently configured feed and URL scope."""
    from app.rss import RssArticle, RssFeedArticle, hashed as feed_hash, load_config
    from app.rss_fetch import CollectionError, public_url
    feeds = {(f.id, feed_hash(f.url)): f for f in load_config()}
    if not feeds:
        return set()
    rows = db.execute(select(RssArticle.id, RssArticle.url, RssArticle.final_url,
        RssFeedArticle.feed_id, RssFeedArticle.feed_url_hash).join(RssFeedArticle,
            RssFeedArticle.article_id == RssArticle.id).where(or_(*[
                and_(RssFeedArticle.feed_id == fid, RssFeedArticle.feed_url_hash == url_hash)
                for fid, url_hash in feeds])))
    allowed = set()
    for article_id, original, final, feed_id, url_hash in rows:
        feed = feeds[(feed_id, url_hash)]
        try:
            public_url(original, feed.hosts)
            public_url(final, feed.hosts)
        except CollectionError:
            continue
        if feed.accepts_path(original) and feed.accepts_path(final):
            allowed.add(article_id)
    return allowed


def candidate_pool(db, run):
    from alphich_kb.personalization.inputs import SelectionScope
    from alphich_kb.personalization.profiles import normalize_company
    from alphich_kb.personalization.reprints import group_reprints
    from alphich_kb.personalization.rules import match_reasons
    from .service import normalized_event
    since = (datetime.fromisoformat(run.as_of) - timedelta(days=90)).isoformat()
    rows = list(db.execute(select(Event, SourceDocument).join(SourceDocument, Event.source_id == SourceDocument.id)
        .where(Event.company_id == run.company_id, SourceDocument.company_id == run.company_id,
            SourceDocument.kind == 'rss_article', SourceDocument.published_at >= since,
            SourceDocument.published_at <= run.as_of, SourceDocument.received_at <= run.as_of)
        .order_by(SourceDocument.received_at.desc(), Event.id).limit(300)))
    profile, candidates = normalize_company(run.profile), []
    allowed_articles = configured_article_ids(db)
    current_sources = {event.id: source for event, source in rows}
    revisions = {ident: hashed([source.id, source.version, source.content_hash])
                 for ident, source in current_sources.items()}
    previously_checked = set()
    # History belongs to the company/role, not a browser session. A refresh or
    # new login must not turn already processed articles into new discoveries.
    history = db.scalars(select(DiscoveryRun).where(DiscoveryRun.company_id == run.company_id,
        DiscoveryRun.role == run.role, DiscoveryRun.id != run.id))
    retryable = {'budget_exceeded', 'discovery_cancelled', 'llm_not_configured'}
    for previous in history:
        if (previous.profile.get('profile_version') != run.profile.get('profile_version') or
                previous.profile.get('payment_context_version') != run.profile.get('payment_context_version')):
            continue
        for old in previous.items:
            ident = old['event_id']
            if ident not in revisions or (old['state'] not in {'completed', 'below_threshold'} and
                    not (old['state'] == 'failed' and old.get('error') not in retryable)):
                continue
            # Old runs did not store this fingerprint. Collector sources at
            # version 1 are immutable; changed source versions remain eligible.
            old_revision = old.get('source_revision')
            if old_revision == revisions[ident] or (old_revision is None and current_sources[ident].version == 1):
                previously_checked.add(ident)
    scope = SelectionScope(run.company_id, run.as_of, frozenset(e.id for e, _ in rows), run.role)
    for event, source in rows:
        if event.id in previously_checked:
            continue
        public = _public_snapshot(db, source)
        if public is None or public['article_id'] not in allowed_articles:
            continue
        item = normalized_event(db, event, source)
        if scope.excludes(item):
            continue
        reasons, _ = match_reasons(profile, item, scope)
        # Macro words alone cannot establish a company-specific channel.
        meaningful = [r for r in reasons if r['tier'] == 'direct' or r['code'] in {
            'unverified_industry_mention', 'unverified_credit_rate_channel', 'unverified_deposit_rate_channel'}]
        if meaningful:
            item['discovery_priority'] = max(r['base'] for r in meaningful)
            item['source_revision'] = revisions[event.id]
            item['age_anchor'] = item['published_at'] or item['available_at']
            candidates.append(item)
    grouped, duplicates = group_reprints(candidates)
    ranked = sorted(grouped, key=lambda e: (-e['discovery_priority'], e['id']))
    return [{'event_id': e['id'], 'title': e['title'], 'state': 'pending', 'grade': None,
             'source_revision': e['source_revision']}
            for e in ranked[:run.max_candidates]], {'scanned': len(rows), 'preselected': len(ranked),
            'duplicates': sum(duplicates.values()), 'pre_filtered': len(rows) - len(candidates),
            'previously_checked': len(previously_checked)}


def enqueue_child(db, run, item, operation, source, now):
    key = hashed([run.id, item['event_id'], operation])
    job = db.scalar(select(Job).where(Job.owner_session_hash == run.owner_session_hash,
        Job.operation == operation, Job.idempotency_key == key))
    if job is None:
        job = Job(id=new_id('job'), owner_session_hash=run.owner_session_hash, company_id=run.company_id,
            operation=operation, idempotency_key=key, request_hash=key, source_id=source.id,
            status='queued', stage='waiting_facts' if operation == 'news_fact_extraction' else 'waiting_analysis',
            result_type='event' if operation == 'news_fact_extraction' else 'insight', attempt_count=0,
            created_at=now, available_at=now)
        db.add(job)
        db.add(AnalysisTask(job_id=job.id, event_id=item['event_id'], role=run.role,
                            as_of=run.as_of, profile=run.profile))
        db.flush()
    item['facts_job_id' if operation == 'news_fact_extraction' else 'analysis_job_id'] = job.id
    item['state'] = 'waiting_facts' if operation == 'news_fact_extraction' else 'waiting_analysis'


def analyze_tick(factory, settings, *, now=None):
    """Advance one run. Paid work is queued one article at a time and never retried here."""
    from alphich_kb.personalization.grading import grade_event
    from alphich_kb.personalization.inputs import SelectionScope
    from alphich_kb.personalization.profiles import normalize_company
    from alphich_kb.personalization.rules import match_reasons
    from .service import normalized_event
    from . import VERSION
    from app.rss_fetch import CollectionError
    now = int(time.time()) if now is None else now
    day = datetime.fromtimestamp(now, timezone.utc).date().isoformat()
    with factory.begin() as db:
        run = db.scalar(select(DiscoveryRun).where(DiscoveryRun.status.in_(ACTIVE))
            .order_by(DiscoveryRun.updated_at, DiscoveryRun.id).with_for_update(skip_locked=True).limit(1))
        if run is None:
            return False
        run.updated_at = now
        items = [dict(i) for i in run.items]
        pending = next((i for i in items if i['state'] in {'waiting_facts', 'waiting_analysis'}), None)
        if pending:
            job = db.get(Job, pending['facts_job_id'] if pending['state'] == 'waiting_facts' else pending['analysis_job_id'])
            if job.status in {'queued', 'running'}:
                if now >= run.deadline_at:
                    run.status, run.stop_reason = 'stopping', 'time_limit'
                return True
            if job.status == 'failed':
                pending.update(state='failed', error=job.error)
                if job.error == 'budget_exceeded' or 'outcome_unknown' in (job.error or ''):
                    write_items(run, items)
                    finish(run, 'partial', job.error)
                    return True
            elif pending['state'] == 'waiting_analysis':
                pending.update(state='completed' if job.result_type == 'insight' else 'below_threshold',
                               insight_id=job.result_id if job.result_type == 'insight' else None)
            else:
                pending['state'] = 'pending'
            write_items(run, items)
        if run.status == 'stopping':
            finish(run, 'partial' if run.stop_reason == 'time_limit' else 'cancelled', run.stop_reason or 'user_cancelled')
            return True
        if now >= run.deadline_at:
            finish(run, 'partial', 'time_limit')
            return True
        company = db.get(Company, run.company_id)
        from .news_index import target_ids
        if company is None or run.company_id not in target_ids():
            finish(run, 'failed', 'company_unavailable')
            return True
        if company.profile_version != run.profile['profile_version']:
            finish(run, 'partial', 'company_profile_changed')
            return True
        if run.status == 'collecting':
            return True
        try:
            settings.require_ready()
        except (AnalysisError, ValueError):
            finish(run, 'partial', 'llm_not_configured')
            return True
        if run.status == 'waiting_ingestion':
            unfinished = db.scalar(select(Job.id).where(Job.company_id == run.company_id,
                Job.operation == 'publication_ingest', Job.status.in_(['queued', 'running']),
                Job.created_at <= run.updated_at).limit(1))
            if unfinished:
                return True
            run.as_of = datetime.fromtimestamp(now, timezone.utc).isoformat()
            try:
                run.items, run.stats = candidate_pool(db, run)
            except CollectionError:
                finish(run, 'failed', 'invalid_rss_config')
                return True
            items = [dict(i) for i in run.items]
            run.status, run.stage = 'analyzing', 'checking_candidates'
        if sum(i['state'] == 'completed' for i in items) >= run.target:
            finish(run, 'completed', 'target_reached')
            return True
        current = next((i for i in items if i['state'] == 'pending'), None)
        if current is None:
            finish(run, 'partial', 'candidate_limit' if len(items) >= run.max_candidates else 'sources_exhausted')
            return True
        if day != run.budget_day:
            finish(run, 'partial', 'budget_day_changed')
            return True
        event = db.get(Event, current['event_id'])
        source = db.get(SourceDocument, event.source_id) if event else None
        if not source or source.company_id != run.company_id or event.company_id != run.company_id:
            current.update(state='failed', error='event_unavailable')
            write_items(run, items)
            return True
        public = _public_snapshot(db, source)
        try:
            allowed_articles = configured_article_ids(db)
        except CollectionError:
            finish(run, 'failed', 'invalid_rss_config')
            return True
        if public is None or public['article_id'] not in allowed_articles:
            current.update(state='failed', error='source_no_longer_configured')
            write_items(run, items)
            return True
        extraction_key = 'extraction_' + hashed([source.id, source.version, source.content_hash, _configuration(settings)])
        extraction = db.get(Extraction, extraction_key)
        if extraction is not None and extraction.status in {'failed', 'running'} and not current.get('facts_job_id'):
            # Unknown previous model outcomes cannot be retried by starting another run.
            current.update(state='failed', error='extraction_unavailable')
            write_items(run, items)
            return True
        if extraction is not None and extraction.status == 'completed':
            normalized = normalized_event(db, event, source, extraction)
            scope = SelectionScope(run.company_id, run.as_of, frozenset({event.id}), run.role)
            profile = normalize_company(run.profile)
            reasons, _ = match_reasons(profile, normalized, scope)
            grade = grade_event(profile, normalized, scope, reasons)
            current.update(grade=grade['relevance_grade'], assessment=grade['relevance_assessment'])
            write_items(run, items)
            if not grade['relevance_selected'] or grade['relevance_grade'] < run.min_grade:
                current['state'] = 'below_threshold'
                write_items(run, items)
                return True
            existing = db.scalar(select(Insight).where(Insight.company_id == run.company_id,
                Insight.role == run.role, Insight.event_id == event.id).order_by(Insight.created_at.desc()).limit(1))
            if (existing and existing.payload.get('profile_version') == run.profile['profile_version']
                    and str(existing.payload.get('event_version')) == str(event.version)
                    and existing.payload.get('relevance_grade', 0) >= run.min_grade
                    and existing.snapshot.get('versions', {}).get('runtime') == VERSION
                    and (settings.allowed_bank_ids is None or all(not e.get('document_id') or
                         e['document_id'] in settings.allowed_bank_ids for e in existing.evidence.values()))
                    and existing.created_at >= now - 86400):
                current.update(state='completed', insight_id=existing.id, reused=True)
                write_items(run, items)
                return True
        budget = db.get(BudgetDay, day)
        limit = min(settings.daily_limit_micros, budget.limit_micros) if budget else settings.daily_limit_micros
        if budget and budget.spent_micros + budget.reserved_micros >= limit:
            if budget.reserved_micros:
                run.stage = 'waiting_for_budget_reservation'
            else:
                finish(run, 'partial', 'budget_exceeded')
            return True
        operation = 'automatic_insight' if current.get('grade') is not None else 'news_fact_extraction'
        enqueue_child(db, run, current, operation, source, now)
        run.stage = 'generating_insight' if operation == 'automatic_insight' else 'extracting_facts'
        write_items(run, items)
    return True
