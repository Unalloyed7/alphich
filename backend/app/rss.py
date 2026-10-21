"""Scheduled RSS collection into the existing company-scoped ingestion queue.

Operator configuration only. Public articles are fetched once; separate company
copies preserve existing ACLs. Collection never creates an LLM analysis task.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import Integer, String, Text, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Mapped, mapped_column

from app.core import Base, Company, make_session_factory
from app.news import Job, PublicationInput, SourceDocument, digest_input, digest_request, new_id
from app.rss_fetch import CollectionError, PublicFetcher, TIMEOUT, public_url
from app.rss_parse import article_text, parse_feed

LOG = logging.getLogger(__name__)
DEFAULT_CONFIG = Path(__file__).with_name('rss_sources.json')


def hashed(value):
    return hashlib.sha256(value.encode()).hexdigest()


def iso(now):
    return datetime.fromtimestamp(now, timezone.utc).isoformat(timespec='seconds')


class FeedConfig(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^[a-z0-9][a-z0-9_-]{0,63}$')
    url: str = Field(max_length=2048)
    article_hosts: list[str] = Field(min_length=1, max_length=10)
    article_path_prefixes: list[str] = Field(default_factory=list, max_length=16)
    poll_seconds: int = Field(default=900, ge=60, le=86400)
    max_items: int = Field(default=10, ge=1, le=50)

    @field_validator('article_hosts')
    @classmethod
    def exact_hosts(cls, values):
        result = []
        for value in values:
            host = value.encode('idna').decode('ascii').lower()
            if not host or '/' in host or ':' in host or '*' in host or host.endswith('.'):
                raise ValueError('Exact DNS hosts required')
            public_url('https://' + host, frozenset({host}))
            result.append(host)
        return sorted(set(result))

    @field_validator('article_path_prefixes')
    @classmethod
    def bounded_paths(cls, values):
        for value in values:
            if (not value.startswith('/') or value.startswith('//') or not value.endswith('/')
                    or len(value) > 200 or any(c.isspace() or c in '?%#\\' for c in value)
                    or any(segment in {'.', '..'} for segment in value.split('/'))):
                raise ValueError('Absolute directory prefixes required, e.g. /business/')
        return sorted(set(values))

    @property
    def hosts(self):
        return frozenset(self.article_hosts)

    def accepts_path(self, url):
        return not self.article_path_prefixes or any(
            urlsplit(url).path.startswith(prefix) for prefix in self.article_path_prefixes)

    @property
    def selection_hash(self):
        return hashed(json.dumps({'url': self.url, 'hosts': sorted(self.hosts),
            'paths': self.article_path_prefixes, 'max_items': self.max_items}, sort_keys=True))


def load_config(path=DEFAULT_CONFIG):
    try:
        raw = json.loads(Path(path).read_text())
        if not isinstance(raw, dict) or set(raw) != {'version', 'feeds'} or raw['version'] != 1:
            raise ValueError()
        feeds = [FeedConfig.model_validate(row) for row in raw['feeds']]
        if not 1 <= len(feeds) <= 10 or len({f.id for f in feeds}) != len(feeds):
            raise ValueError()
        for feed in feeds:
            public_url(feed.url, feed.hosts)
        return feeds
    except (ValueError, OSError, TypeError, KeyError, CollectionError):
        raise CollectionError('invalid_rss_config') from None


class FeedState(Base):
    __tablename__ = 'rss_feed_states'
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    url: Mapped[str] = mapped_column(String(2048))
    etag: Mapped[str | None] = mapped_column(String(500))
    last_modified: Mapped[str | None] = mapped_column(String(500))
    next_poll_at: Mapped[int] = mapped_column(Integer, default=0)
    lease_token: Mapped[str | None] = mapped_column(String(40))
    lease_until: Mapped[int] = mapped_column(Integer, default=0)
    last_checked_at: Mapped[int | None] = mapped_column(Integer)
    last_success_at: Mapped[int | None] = mapped_column(Integer)
    failures: Mapped[int] = mapped_column(Integer, default=0)
    last_error: Mapped[str | None] = mapped_column(String(80))


class RssArticle(Base):
    __tablename__ = 'rss_articles'
    id: Mapped[str] = mapped_column(String(64), primary_key=True)  # Hash of first canonical RSS link.
    first_feed_id: Mapped[str] = mapped_column(String(64), index=True)
    url: Mapped[str] = mapped_column(String(2048))
    final_url: Mapped[str] = mapped_column(String(2048))
    title: Mapped[str] = mapped_column(String(300))
    body: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    published_at: Mapped[str | None] = mapped_column(String(40))
    fetched_at: Mapped[str] = mapped_column(String(40))


class RssDelivery(Base):
    __tablename__ = 'rss_deliveries'
    article_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(40))
    job_id: Mapped[str] = mapped_column(String(40))


class RssFeedArticle(Base):
    __tablename__ = 'rss_feed_articles'
    feed_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    feed_url_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    article_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    seen_at: Mapped[int] = mapped_column(Integer)


class FeedSelection(Base):
    """Additive selection state, including empty feeds; no ALTER of existing feed states."""
    __tablename__ = 'rss_feed_selections'
    feed_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    configuration_hash: Mapped[str] = mapped_column(String(64))


def target_companies(explicit=None):
    allowed = {s.strip() for s in os.getenv('DEMO_COMPANY_IDS', '').split(',') if s.strip()}
    targets = set(explicit or [s.strip() for s in os.getenv('RSS_COMPANY_IDS', '').split(',') if s.strip()] or allowed)
    if not targets or not targets <= allowed or len(targets) > 20:
        raise CollectionError('rss_targets_not_allowed')
    return sorted(targets)


def _claim(factory, feed, now, force):
    token = new_id('rss')
    with factory.begin() as db:
        if db.get(FeedState, feed.id) is None:
            try:
                with db.begin_nested():
                    db.add(FeedState(id=feed.id, url=feed.url, next_poll_at=0, lease_until=0, failures=0))
                    db.flush()
            except IntegrityError:
                pass
        changed = db.execute(update(FeedState).where(FeedState.id == feed.id, FeedState.lease_until <= now,
            True if force else FeedState.next_poll_at <= now).values(
                lease_token=token, lease_until=now + (feed.max_items + 1) * TIMEOUT + 90,
                last_checked_at=now))
        if changed.rowcount != 1:
            return None
        state = db.get(FeedState, feed.id)
        selection = db.get(FeedSelection, feed.id)
        same_selection = state.url == feed.url and selection is not None and selection.configuration_hash == feed.selection_hash
        validators = {'If-None-Match': state.etag, 'If-Modified-Since': state.last_modified} if same_selection else {}
    return token, validators


def _active(db, feed, token, now):
    state = db.scalar(select(FeedState).where(FeedState.id == feed.id).with_for_update())
    if state is None or state.lease_token != token or state.lease_until <= now:
        raise CollectionError('rss_lease_lost')
    return state


def _store_article(factory, feed, token, item, url, downloaded, now):
    article_id = hashed(url)
    body = article_text(downloaded.body, downloaded.url, downloaded.headers.get('content-type', ''))
    payload = PublicationInput(title=item.title, text=body, source_url=downloaded.url, published_at=item.published_at)
    with factory.begin() as db:
        _active(db, feed, token, now)
        if db.get(RssArticle, article_id) is None:
            try:
                with db.begin_nested():
                    db.add(RssArticle(id=article_id, first_feed_id=feed.id, url=url, final_url=downloaded.url,
                        title=payload.title, body=payload.text, content_hash=digest_input(payload),
                        published_at=payload.published_at.isoformat() if payload.published_at else None, fetched_at=iso(now)))
                    db.flush()
            except IntegrityError:
                if db.get(RssArticle, article_id) is None:
                    raise
    return article_id


def _deliver(factory, feed, token, article_id, targets, now):
    created = 0
    with factory.begin() as db:
        _active(db, feed, token, now)
        article = db.scalar(select(RssArticle).where(RssArticle.id == article_id).with_for_update())
        public_url(article.final_url, feed.hosts)
        if not feed.accepts_path(article.url) or not feed.accepts_path(article.final_url):
            raise CollectionError('article_path_not_allowed')
        membership_key = (feed.id, hashed(feed.url), article_id)
        if db.get(RssFeedArticle, membership_key) is None:
            db.add(RssFeedArticle(feed_id=feed.id, feed_url_hash=hashed(feed.url), article_id=article_id, seen_at=now))
        for company_id in targets:
            if db.get(RssDelivery, (article_id, company_id)) is not None:
                continue
            if db.get(Company, company_id) is None:
                raise CollectionError('rss_company_not_imported')
            # Existing tenant boundaries are unchanged; no private publication is made public.
            source = db.scalar(select(SourceDocument).where(SourceDocument.company_id == company_id,
                SourceDocument.content_hash == article.content_hash).with_for_update())
            if source is None:
                source = SourceDocument(id=new_id('src'), company_id=company_id, kind='rss_article',
                    title=article.title, body=article.body, source_url=article.final_url,
                    published_at=article.published_at, received_at=iso(now), fetched_at=article.fetched_at,
                    content_hash=article.content_hash, extraction_status='raw_only',
                    completeness='extracted_article_text', version=1)
                try:
                    with db.begin_nested():
                        db.add(source)
                        db.flush()
                except IntegrityError:
                    source = db.scalar(select(SourceDocument).where(SourceDocument.company_id == company_id,
                        SourceDocument.content_hash == article.content_hash).with_for_update())
                    if source is None:
                        raise
            # Public background ingestion has no demo-session token; jobs are operational,
            # while completed events are visible through the normal company events API.
            owner = hashed('rss-system:' + company_id)
            key = 'rss:' + source.id
            job = db.scalar(select(Job).where(Job.owner_session_hash == owner,
                Job.operation == 'publication_ingest', Job.idempotency_key == key))
            if job is None:
                payload = PublicationInput(title=source.title, text=source.body,
                    source_url=source.source_url, published_at=source.published_at)
                job = Job(id=new_id('job'), owner_session_hash=owner, company_id=company_id,
                    operation='publication_ingest', idempotency_key=key, request_hash=digest_request(payload),
                    source_id=source.id, status='queued', stage='waiting', result_type='event',
                    attempt_count=0, created_at=now, available_at=now)
                db.add(job)
                db.flush()
            db.add(RssDelivery(article_id=article_id, company_id=company_id, source_id=source.id, job_id=job.id))
            created += 1
    return created


def poll_feed(factory, feed, targets, *, fetcher=None, clock=time.time, force=False):
    now = int(clock())
    claimed = _claim(factory, feed, now, force)
    result = {'feed_id': feed.id, 'status': 'not_due', 'downloaded': 0, 'queued': 0, 'reused': 0,
              'feed_items': 0, 'filtered': 0, 'rejected': 0, 'errors': []}
    if claimed is None:
        return result
    token, validators = claimed
    fetcher = fetcher or PublicFetcher()
    response = None
    try:
        response = fetcher.get(feed.url, feed.hosts, validators)
        if response.status == 304:
            # Newly configured companies still receive cached public articles on an unchanged feed.
            with factory() as db:
                cached_rows = list(db.execute(select(RssArticle.id, RssArticle.url, RssArticle.final_url)
                    .join(RssFeedArticle, RssFeedArticle.article_id == RssArticle.id).where(
                        RssFeedArticle.feed_id == feed.id, RssFeedArticle.feed_url_hash == hashed(feed.url))
                    .order_by(RssFeedArticle.seen_at.desc(), RssFeedArticle.article_id).limit(1000)))
            for article_id, original, final in cached_rows:
                try:
                    public_url(original, feed.hosts)
                    public_url(final, feed.hosts)
                except CollectionError:
                    result['rejected'] += 1
                    continue
                if not feed.accepts_path(original) or not feed.accepts_path(final):
                    result['filtered'] += 1
                    continue
                result['queued'] += _deliver(factory, feed, token, article_id, targets, int(clock()))
                result['reused'] += 1
                if result['reused'] >= feed.max_items:
                    break
            result['status'] = 'not_modified'
        elif response.status == 200:
            items = parse_feed(response.body, response.url)
            result['feed_items'] = len(items)
            seen = set()
            for item in items:
                try:
                    url = public_url(item.url, feed.hosts)
                except CollectionError:
                    result['rejected'] += 1
                    continue
                article_id = hashed(url)
                if not feed.accepts_path(url):
                    result['filtered'] += 1
                    continue
                if article_id in seen:
                    continue
                if len(seen) >= feed.max_items:
                    break
                seen.add(article_id)
                try:
                    with factory() as db:
                        cached = db.get(RssArticle, article_id)
                    if cached:
                        public_url(cached.final_url, feed.hosts)
                        if not feed.accepts_path(cached.final_url):
                            result['filtered'] += 1
                            continue
                        result['reused'] += 1
                    else:
                        downloaded = fetcher.get(url, feed.hosts)
                        if downloaded.status != 200:
                            raise CollectionError('article_http_' + str(downloaded.status))
                        public_url(downloaded.url, feed.hosts)
                        if not feed.accepts_path(downloaded.url):
                            result['filtered'] += 1
                            continue
                        article_id = _store_article(factory, feed, token, item, url, downloaded, int(clock()))
                        result['downloaded'] += 1
                    result['queued'] += _deliver(factory, feed, token, article_id, targets, int(clock()))
                except CollectionError as error:
                    if error.code == 'rss_lease_lost':
                        raise
                    result['errors'].append(error.code)
            result['status'] = 'partial' if result['errors'] else 'collected'
        else:
            raise CollectionError('feed_http_' + str(response.status))
    except CollectionError as error:
        result['status'] = 'failed'
        result['errors'].append(error.code)
    except Exception:
        # Do not log article text, query strings, credentials or raw parser/network errors.
        result['status'] = 'failed'
        result['errors'].append('rss_internal_error')
    finally:
        now = int(clock())
        with factory.begin() as db:
            state = db.scalar(select(FeedState).where(FeedState.id == feed.id).with_for_update())
            if state.lease_token == token:
                failed = bool(result['errors'])
                state.failures = state.failures + 1 if failed else 0
                state.last_error = result['errors'][0] if failed else None
                state.next_poll_at = now + (min(feed.poll_seconds, 30 * 2 ** min(state.failures, 10)) if failed else feed.poll_seconds)
                state.lease_token, state.lease_until = None, 0
                # A partial feed must be downloaded again: an ETag must not hide failed articles.
                if failed:
                    state.etag = state.last_modified = None
                elif response is not None:
                    selection = db.get(FeedSelection, feed.id)
                    unchanged = (response.status == 304 and state.url == feed.url and selection is not None
                                 and selection.configuration_hash == feed.selection_hash)
                    state.etag = response.headers.get('etag', state.etag if unchanged else '')
                    state.last_modified = response.headers.get('last-modified', state.last_modified if unchanged else '')
                    state.etag = (state.etag or '')[:500] or None
                    state.last_modified = (state.last_modified or '')[:500] or None
                state.url = feed.url
                selection = db.get(FeedSelection, feed.id)
                if selection is None:
                    db.add(FeedSelection(feed_id=feed.id, configuration_hash=feed.selection_hash))
                else:
                    selection.configuration_hash = feed.selection_hash
                if not failed:
                    state.last_success_at = now
    return result


def collect_once(factory, feeds, targets, **kwargs):
    # Validate the entire bounded fan-out before making any outbound request.
    allowed = target_companies(targets)
    with factory() as db:
        present = set(db.scalars(select(Company.id).where(Company.id.in_(allowed))))
    if present != set(allowed):
        return [{'status': 'waiting_for_company_import', 'missing_companies': sorted(set(allowed) - present)}]
    return [poll_feed(factory, feed, allowed, **kwargs) for feed in feeds]


def main():
    parser = argparse.ArgumentParser(description='Collect configured RSS/Atom news without LLM calls')
    parser.add_argument('--config', default=os.getenv('RSS_CONFIG', str(DEFAULT_CONFIG)))
    parser.add_argument('--company-id', action='append')
    parser.add_argument('--once', action='store_true', help='Collect due feeds once and exit')
    parser.add_argument('--force', action='store_true', help='Ignore poll time for --once; active leases remain enforced')
    parser.add_argument('--status', action='store_true', help='Read collection state without HTTP requests')
    args = parser.parse_args()
    if args.force and not args.once:
        parser.error('--force requires --once')
    logging.basicConfig(level=logging.INFO)
    try:
        feeds = load_config(args.config)
        targets = [] if args.status else target_companies(args.company_id)
    except CollectionError as error:
        print(json.dumps({'status': 'failed', 'error': error.code}))
        return 2
    engine, factory = make_session_factory()
    Base.metadata.create_all(engine)
    try:
        if args.status:
            with factory() as db:
                print(json.dumps([{'feed_id': feed.id, 'configured_url': feed.url,
                    'last_success_at': state.last_success_at if state else None,
                    'next_poll_at': state.next_poll_at if state else None,
                    'error': state.last_error if state else None}
                    for feed in feeds for state in [db.get(FeedState, feed.id)]], ensure_ascii=False))
            return 0
        while True:
            report = collect_once(factory, feeds, targets, force=args.force)
            print(json.dumps(report, ensure_ascii=False), flush=True)
            if args.once:
                return 2 if any(r['status'] in {'failed', 'partial', 'waiting_for_company_import'} for r in report) else 0
            time.sleep(15)
    except KeyboardInterrupt:
        return 0
    finally:
        engine.dispose()


if __name__ == '__main__':
    raise SystemExit(main())
