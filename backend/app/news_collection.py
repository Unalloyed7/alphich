"""Durable, unpaid random sampling of configured public news sources.

The API creates work only. HTTP collection and ordinary publication ingestion
are handled by the existing worker, with no dependency on LLM configuration.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import random
import re
import time

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import JSON, Integer, String, UniqueConstraint, case, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from app.core import Base, Company
from app.news import Event, Job, new_id

ACTIVE_STATUSES = ('queued', 'collecting', 'waiting_ingestion')
RUN_SECONDS = 15 * 60
START_COOLDOWN_SECONDS = 60


class NewsCollectionRun(Base):
    __tablename__ = 'news_collection_runs'
    __table_args__ = (UniqueConstraint('owner_session_hash', 'idempotency_key'),)
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    owner_session_hash: Mapped[str] = mapped_column(String(64), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(128))
    request_hash: Mapped[str] = mapped_column(String(64))
    active_key: Mapped[str | None] = mapped_column(String(100), unique=True)
    status: Mapped[str] = mapped_column(String(32))
    requested_count: Mapped[int] = mapped_column(Integer)
    feed_ids: Mapped[list] = mapped_column(JSON, default=list)
    sample_seed: Mapped[str] = mapped_column(String(64))
    checked_sources: Mapped[int] = mapped_column(Integer, default=0)
    items: Mapped[list] = mapped_column(JSON, default=list)
    errors: Mapped[list] = mapped_column(JSON, default=list)
    newly_downloaded: Mapped[int] = mapped_column(Integer, default=0)
    reused: Mapped[int] = mapped_column(Integer, default=0)
    stop_reason: Mapped[str | None] = mapped_column(String(80))
    lease_token: Mapped[str | None] = mapped_column(String(40))
    lease_until: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[int] = mapped_column(Integer)
    updated_at: Mapped[int] = mapped_column(Integer)
    deadline_at: Mapped[int] = mapped_column(Integer)


class NewsCollectionRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    count: int = Field(default=10, ge=1, le=100, strict=True)


def _finish(run, status, reason, now):
    run.status, run.stop_reason = status, reason
    run.active_key, run.lease_token, run.lease_until = None, None, 0
    run.updated_at = now


def _add_errors(run, errors):
    # Exception messages and remote response bodies never reach the browser.
    safe = [str(code) for code in errors if re.fullmatch(r'[a-z0-9_-]{1,80}', str(code))]
    run.errors = list(dict.fromkeys([*(run.errors or []), *safe]))[:100]


def view_collection(db, run):
    items = [dict(item) for item in run.items or []]
    job_ids = [item['job_id'] for item in items]
    jobs = {job.id: job for job in db.scalars(select(Job).where(Job.id.in_(job_ids)))} if job_ids else {}
    for item in items:
        job = jobs.get(item['job_id'])
        item['event_id'] = job.result_id if job and job.status == 'succeeded' else None
    return {
        'id': run.id, 'company_id': run.company_id, 'status': run.status,
        'requested_count': run.requested_count, 'collected_count': len(items),
        'checked_sources': run.checked_sources, 'total_sources': len(run.feed_ids),
        'created_at': run.created_at, 'updated_at': run.updated_at, 'deadline_at': run.deadline_at,
        'stop_reason': run.stop_reason, 'errors': run.errors or [], 'items': items,
        'newly_downloaded': run.newly_downloaded, 'reused': run.reused,
    }


def register_collection_routes(app, get_db, current_session):
    def own(run, demo):
        if run is None or run.company_id != demo.company_id or run.owner_session_hash != demo.token_hash:
            raise HTTPException(404, 'News search not found')
        return run

    @app.get('/api/v1/companies/{company_id}/news-search')
    def latest_collection(company_id: str, demo=Depends(current_session), db=Depends(get_db)):
        if company_id != demo.company_id:
            raise HTTPException(403, 'Session does not grant access to this company')
        run = db.scalar(select(NewsCollectionRun).where(NewsCollectionRun.company_id == company_id,
            NewsCollectionRun.owner_session_hash == demo.token_hash)
            .order_by(NewsCollectionRun.created_at.desc(), NewsCollectionRun.id.desc()).limit(1))
        return {'run': view_collection(db, run) if run else None}

    @app.post('/api/v1/companies/{company_id}/news-search', status_code=202)
    def start_collection(company_id: str, payload: NewsCollectionRequest,
                         idempotency_key: str | None = Header(None),
                         demo=Depends(current_session), db=Depends(get_db)):
        if company_id != demo.company_id:
            raise HTTPException(403, 'Session does not grant access to this company')
        if not idempotency_key or not re.fullmatch(r'[A-Za-z0-9._:-]{8,128}', idempotency_key):
            raise HTTPException(400, 'A valid Idempotency-Key is required')
        fingerprint = hashlib.sha256(json.dumps(payload.model_dump(), sort_keys=True).encode()).hexdigest()
        previous = db.scalar(select(NewsCollectionRun).where(
            NewsCollectionRun.owner_session_hash == demo.token_hash,
            NewsCollectionRun.idempotency_key == idempotency_key))
        if previous:
            if previous.request_hash != fingerprint:
                raise HTTPException(409, 'Idempotency-Key was already used with a different payload')
            return view_collection(db, previous)
        # Lock the company to serialize cooldown checks, including after the
        # previous active row has become terminal. The unique active key is the
        # additional database safeguard for concurrent starts.
        company = db.scalar(select(Company).where(Company.id == company_id).with_for_update())
        if company is None:
            raise HTTPException(404, 'Company not found')
        now = int(time.time())
        active = db.scalar(select(NewsCollectionRun).where(NewsCollectionRun.active_key == company_id))
        if active and active.deadline_at <= now and active.lease_until <= now:
            _finish(active, 'partial' if active.items else 'failed', 'time_limit', now)
            db.flush()
            active = None
        if active:
            if active.owner_session_hash == demo.token_hash:
                return view_collection(db, active)
            raise HTTPException(409, 'A news search for this company is already running')
        latest = db.scalar(select(NewsCollectionRun).where(NewsCollectionRun.company_id == company_id)
            .order_by(NewsCollectionRun.created_at.desc()).limit(1))
        if latest and latest.created_at + START_COOLDOWN_SECONDS > now:
            raise HTTPException(429, 'Wait before starting another news search',
                                headers={'Retry-After': str(latest.created_at + START_COOLDOWN_SECONDS - now)})
        from app.rss import load_config
        from app.rss_fetch import CollectionError
        try:
            feed_ids = [feed.id for feed in load_config()]
        except (CollectionError, ValueError):
            raise HTTPException(503, 'invalid_rss_config') from None
        random.SystemRandom().shuffle(feed_ids)
        run = NewsCollectionRun(id=new_id('news'), company_id=company_id, owner_session_hash=demo.token_hash,
            idempotency_key=idempotency_key, request_hash=fingerprint, active_key=company_id, status='queued',
            requested_count=payload.count, feed_ids=feed_ids, sample_seed=new_id('sample'), checked_sources=0,
            items=[], errors=[], newly_downloaded=0, reused=0, lease_until=0,
            created_at=now, updated_at=now, deadline_at=now + RUN_SECONDS)
        try:
            with db.begin_nested():
                db.add(run)
                db.flush()
        except IntegrityError:
            raise HTTPException(409, 'A concurrent news search was started; refresh progress') from None
        db.commit()
        return view_collection(db, run)

    @app.get('/api/v1/news-search/{run_id}')
    def collection_detail(run_id: str, demo=Depends(current_session), db=Depends(get_db)):
        return view_collection(db, own(db.get(NewsCollectionRun, run_id), demo))

    @app.post('/api/v1/news-search/{run_id}/stop')
    def stop_collection(run_id: str, demo=Depends(current_session), db=Depends(get_db)):
        run = own(db.scalar(select(NewsCollectionRun).where(NewsCollectionRun.id == run_id).with_for_update()), demo)
        if run.status in ACTIVE_STATUSES:
            _finish(run, 'cancelled', 'user_cancelled', int(time.time()))
            db.commit()
        return view_collection(db, run)


def _not_future(article, now):
    if article.published_at is None:
        return True
    try:
        published = datetime.fromisoformat(article.published_at.replace('Z', '+00:00'))
        return published.tzinfo is not None and published.timestamp() <= now
    except (ValueError, OverflowError):
        return False


def _cached_candidates(factory, feed, excluded, seed, now):
    from app.rss import RssArticle, RssFeedArticle, hashed
    from app.rss_fetch import CollectionError, public_url
    with factory() as db:
        rows = list(db.scalars(select(RssArticle).join(RssFeedArticle,
            RssFeedArticle.article_id == RssArticle.id).where(RssFeedArticle.feed_id == feed.id,
                RssFeedArticle.feed_url_hash == hashed(feed.url))
            .order_by(RssFeedArticle.seen_at.desc(), RssArticle.id).limit(1000)))
    random.Random(seed + ':' + feed.id).shuffle(rows)
    accepted = []
    for article in rows:
        if article.id in excluded or not _not_future(article, now):
            continue
        try:
            public_url(article.url, feed.hosts)
            public_url(article.final_url, feed.hosts)
        except CollectionError:
            continue
        if feed.accepts_path(article.url) and feed.accepts_path(article.final_url):
            accepted.append(article.id)
    return accepted


def _items_for(factory, article_ids, company_id, now):
    from app.rss import RssArticle, RssDelivery
    with factory() as db:
        rows = db.execute(select(RssArticle, RssDelivery).join(RssDelivery,
            RssDelivery.article_id == RssArticle.id).where(RssArticle.id.in_(article_ids),
                RssDelivery.company_id == company_id))
        by_id = {article.id: {'article_id': article.id, 'source_id': delivery.source_id,
            'job_id': delivery.job_id, 'event_id': None, 'title': article.title,
            'url': article.final_url, 'published_at': article.published_at}
            for article, delivery in rows if _not_future(article, now)}
    return [by_id[ident] for ident in article_ids if ident in by_id]


def collection_tick(factory, *, feeds=None, poll=None, enqueue=None, now=None):
    """Advance one durable unpaid search by one source or ingestion check."""
    from app.rss import enqueue_cached, load_config, poll_feed, target_companies
    from app.rss_fetch import CollectionError
    now = int(time.time()) if now is None else int(now)
    with factory.begin() as db:
        run = db.scalar(select(NewsCollectionRun).where(NewsCollectionRun.status.in_(ACTIVE_STATUSES),
            NewsCollectionRun.lease_until <= now).order_by(
                case((NewsCollectionRun.status == 'waiting_ingestion', 1), else_=0),
                NewsCollectionRun.updated_at, NewsCollectionRun.id)
            .with_for_update(skip_locked=True).limit(1))
        if run is None:
            return False
        if run.deadline_at <= now:
            _finish(run, 'partial' if run.items else 'failed', 'time_limit', now)
            return True
        if run.status == 'waiting_ingestion':
            jobs = {job.id: job for job in db.scalars(select(Job).where(
                Job.id.in_([item['job_id'] for item in run.items])))}
            if any(job.status in {'queued', 'running'} for job in jobs.values()):
                return False
            failed = len(jobs) != len(run.items) or any(job.status != 'succeeded' for job in jobs.values())
            if failed:
                _add_errors(run, ['ingestion_failed'])
            enough = len(run.items) >= run.requested_count and not failed
            _finish(run, 'succeeded' if enough else 'partial',
                    'target_reached' if enough else 'ingestion_failed' if failed else 'sources_exhausted', now)
            return True
        try:
            configured = {feed.id: feed for feed in (load_config() if feeds is None else feeds)}
            target_companies([run.company_id])
        except (CollectionError, ValueError):
            _finish(run, 'failed', 'invalid_rss_config', now)
            return True
        if run.checked_sources >= len(run.feed_ids):
            if run.items:
                run.status = 'waiting_ingestion'
            else:
                _finish(run, 'partial', 'sources_exhausted', now)
            return True
        feed_id = run.feed_ids[run.checked_sources]
        if feed_id not in configured:
            _add_errors(run, ['source_configuration_changed'])
            run.checked_sources += 1
            run.updated_at = now
            return True
        feed = configured[feed_id]
        token = new_id('lease')
        changed = db.execute(update(NewsCollectionRun).where(NewsCollectionRun.id == run.id,
            NewsCollectionRun.lease_until <= now, NewsCollectionRun.status.in_(ACTIVE_STATUSES))
            .values(lease_token=token, lease_until=now + (feed.max_items + 1) * 25 + 120,
                    status='collecting', updated_at=now))
        if changed.rowcount != 1:
            return False
        history = db.scalars(select(NewsCollectionRun).where(NewsCollectionRun.company_id == run.company_id))
        previous_items = [item for previous in history for item in previous.items or []]
        excluded = frozenset(item['article_id'] for item in previous_items)
        excluded_sources = {item['source_id'] for item in previous_items}
        run_id, company_id, seed = run.id, run.company_id, run.sample_seed
        remaining = run.requested_count - len(run.items)
    report, ids, errors = {}, [], []
    try:
        report = (poll or poll_feed)(factory, feed, [company_id], force=True,
            sample_seed=seed, max_articles=remaining, exclude_article_ids=excluded,
            force_cooldown_seconds=START_COOLDOWN_SECONDS)
        ids = list(dict.fromkeys(report.get('article_ids', [])))
        errors.extend(report.get('errors', []))
        # A recent background poll/304 may already have populated safe cached
        # snapshots. They are useful test material, but never repeated between
        # this company's random-search runs.
        selected = set(ids) | set(excluded)
        for article_id in _cached_candidates(factory, feed, selected, seed, now):
            if len(ids) >= remaining:
                break
            try:
                (enqueue or enqueue_cached)(factory, feed, article_id, [company_id], now=now)
                ids.append(article_id)
            except Exception:
                errors.append('cached_article_failed')
        ids = [ident for ident in ids if ident not in excluded][:remaining]
    except Exception:
        errors.append('collection_failed')
    # Retain successfully enqueued publications even if another source/cache item failed.
    items = []
    for item in _items_for(factory, ids, company_id, now):
        if item['source_id'] not in excluded_sources:
            items.append(item)
            excluded_sources.add(item['source_id'])
    with factory.begin() as db:
        run = db.scalar(select(NewsCollectionRun).where(NewsCollectionRun.id == run_id).with_for_update())
        if run is None or run.lease_token != token or run.status not in ACTIVE_STATUSES:
            return True
        run.items = [*(run.items or []), *items]
        run.newly_downloaded += min(max(int(report.get('downloaded', 0)), 0), len(items))
        run.reused += len(items) - min(max(int(report.get('downloaded', 0)), 0), len(items))
        _add_errors(run, errors)
        run.checked_sources += 1
        run.lease_token, run.lease_until, run.updated_at = None, 0, now
        if len(run.items) >= run.requested_count or run.checked_sources >= len(run.feed_ids):
            if run.items:
                run.status = 'waiting_ingestion'
            else:
                _finish(run, 'partial', 'sources_exhausted', now)
    return True
