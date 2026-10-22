"""Continuous RSS scheduling, truthful status and unpaid random collection policies."""
from concurrent.futures import Future
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import time

import pytest
from sqlalchemy import func, select

from app.news import Event, Job, SourceDocument
from app.rss import (CollectorHeartbeat, FeedConfig, FeedState, RssArticle, RssDelivery,
                     RssSourcePolicy, enqueue_cached, load_config, poll_feed,
                     scheduler_tick, touch_heartbeat)
from app.rss_fetch import CollectionError, Download
from app.worker import process_next_job
from app.intelligence.automation import AutomationConfig, AutomationRecord, dispatch_once
from app.intelligence.storage import LLMCall
from test_intelligence import store
from test_news import app_store
from test_news_collection import FEED, SampleFetcher


@pytest.mark.parametrize('value, seconds', [('1', 60), ('5', 300), ('1440', 86400)])
def test_poll_minutes_override_bounds(monkeypatch, value, seconds):
    monkeypatch.setenv('RSS_POLL_MINUTES', value)
    feeds = load_config()
    assert feeds and {feed.poll_seconds for feed in feeds} == {seconds}


@pytest.mark.parametrize('value', ['0', '-1', '1441', '1.5', 'five'])
def test_invalid_poll_override_fails_before_collection(monkeypatch, value):
    monkeypatch.setenv('RSS_POLL_MINUTES', value)
    with pytest.raises(CollectionError, match='invalid_rss_config'):
        load_config()


def test_blank_poll_override_preserves_per_source_intervals(monkeypatch):
    monkeypatch.delenv('RSS_POLL_MINUTES', raising=False)
    original = {feed.id: feed.poll_seconds for feed in load_config()}
    monkeypatch.setenv('RSS_POLL_MINUTES', '  ')
    assert {feed.id: feed.poll_seconds for feed in load_config()} == original
    assert len(set(original.values())) > 1


class PendingExecutor:
    def __init__(self):
        self.submitted = []

    def submit(self, function, *args):
        future = Future()
        self.submitted.append((function, args, future))
        return future


def test_scheduler_keeps_slow_source_independent_and_respects_due_time_lease_and_disabled(app_store):
    _, factory, _ = app_store
    feeds = [FEED.model_copy(update={'id': name}) for name in ('slow', 'fast', 'later', 'leased')]
    slow = Future()
    inflight, executor = {'slow': slow}, PendingExecutor()
    with factory.begin() as db:
        for ident, due, lease in [('fast', 0, 0), ('later', 2000, 0), ('leased', 0, 2000)]:
            db.add(FeedState(id=ident, url=FEED.url, next_poll_at=due, lease_until=lease, failures=0))
    assert scheduler_tick(factory, feeds, ['U_A'], executor, inflight, now=1000) == []
    assert len(executor.submitted) == 1
    assert executor.submitted[0][1][1][0].id == 'fast'
    assert set(inflight) == {'slow', 'fast'}
    fast = executor.submitted[0][2]
    fast.set_result([{'feed_id': 'fast', 'status': 'collected'}])
    with factory.begin() as db:
        db.get(FeedState, 'fast').next_poll_at = 2000
    assert scheduler_tick(factory, feeds, ['U_A'], executor, inflight, now=1001) == [
        {'feed_id': 'fast', 'status': 'collected'}]
    assert len(executor.submitted) == 1 and not slow.done()
    with factory() as db:
        heartbeat = db.get(CollectorHeartbeat, 'main')
        assert heartbeat.enabled is True and heartbeat.seen_at == 1001
    assert scheduler_tick(factory, feeds, ['U_A'], executor, inflight, now=3000, enabled=False) == []
    assert len(executor.submitted) == 1
    with factory() as db:
        assert db.get(CollectorHeartbeat, 'main').enabled is False
    slow.set_exception(RuntimeError('private network exception must not leak'))
    reports = scheduler_tick(factory, feeds, ['U_A'], executor, inflight, now=3001, enabled=False)
    assert reports == [{'feed_id': 'slow', 'status': 'failed', 'errors': ['rss_internal_error']}]


def test_status_requires_session_and_distinguishes_unknown_running_stale_disabled(app_store, monkeypatch):
    client, factory, headers = app_store
    monkeypatch.setenv('RSS_ENABLED', '1')
    monkeypatch.setenv('RSS_POLL_MINUTES', '3')
    path = '/api/v1/news-collection/status'
    assert client.get(path).status_code == 401
    unknown = client.get(path, headers=headers['U_A'])
    assert unknown.status_code == 200, unknown.text
    assert unknown.json()['state'] == 'unknown' and unknown.json()['poll_minutes'] == 3
    assert {feed['poll_seconds'] for feed in unknown.json()['feeds']} == {180}
    touch_heartbeat(factory)
    assert client.get(path, headers=headers['U_A']).json()['state'] == 'running'
    touch_heartbeat(factory, now=int(time.time()) - 120)
    assert client.get(path, headers=headers['U_A']).json()['state'] == 'stale'
    monkeypatch.setenv('RSS_ENABLED', '0')
    assert client.get(path, headers=headers['U_A']).json()['state'] == 'disabled'


def test_cached_enqueue_requires_current_feed_membership_and_deduplicates_jobs(app_store):
    _, factory, _ = app_store
    report = poll_feed(factory, FEED, ['U_A'], fetcher=SampleFetcher(count=2), force=True)
    ident = report['article_ids'][0]
    assert enqueue_cached(factory, FEED, ident, ['U_A']) == 0
    assert enqueue_cached(factory, FEED, ident, ['U_B']) == 1
    assert enqueue_cached(factory, FEED, ident, ['U_B']) == 0
    changed_feed = FEED.model_copy(update={'url': FEED.url + '/changed'})
    with pytest.raises(CollectionError, match='article_not_in_configured_feed'):
        enqueue_cached(factory, changed_feed, ident, ['U_A'])
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 3
        assert db.scalar(select(func.count()).select_from(RssDelivery)) == 3


def test_random_poll_honors_seed_limit_exclusions_and_clears_conditional_listing(app_store):
    _, factory, _ = app_store
    fetcher = SampleFetcher(count=8)
    first = poll_feed(factory, FEED, ['U_A'], fetcher=fetcher, force=True,
                      sample_seed='repeatable-order', max_articles=2)
    assert len(first['article_ids']) == first['downloaded'] == 2
    second = poll_feed(factory, FEED, ['U_A'], fetcher=fetcher, force=True,
        sample_seed='repeatable-order', max_articles=2, exclude_article_ids=frozenset(first['article_ids']))
    assert len(second['article_ids']) == second['downloaded'] == 2
    assert not set(first['article_ids']) & set(second['article_ids'])
    with factory() as db:
        assert db.get(FeedState, FEED.id).etag is None
        assert db.scalar(select(func.count()).select_from(RssSourcePolicy)) == 4


def test_cached_prefix_does_not_consume_new_download_quota_or_hide_backlog_with_etag(app_store):
    _, factory, _ = app_store
    feed = FEED.model_copy(update={'max_items': 1})
    fetcher = SampleFetcher(count=3)
    first = poll_feed(factory, feed, ['U_A'], fetcher=fetcher, force=True)
    assert first['downloaded'] == 1
    with factory() as db:
        assert db.get(FeedState, feed.id).etag is None
    second = poll_feed(factory, feed, ['U_A'], fetcher=fetcher, force=True)
    assert second['downloaded'] == second['reused'] == 1
    with factory() as db:
        assert db.get(FeedState, feed.id).etag is None
    third = poll_feed(factory, feed, ['U_A'], fetcher=fetcher, force=True)
    assert third['downloaded'] == 1 and third['reused'] == 2
    with factory() as db:
        assert db.get(FeedState, feed.id).etag == 'sample-v1'
        assert db.scalar(select(func.count()).select_from(RssArticle)) == 3


def test_new_test_source_suppresses_automatic_ai_without_altering_existing_normal_source(store, monkeypatch):
    _, factory, _, settings = store
    class RecentFetcher(SampleFetcher):
        def get(self, url, hosts, validators=None):
            response = super().get(url, hosts, validators)
            if url == FEED.url:
                stamp = format_datetime(datetime.now(timezone.utc) - timedelta(minutes=5)).encode()
                return Download(response.url, response.status, response.headers,
                    response.body.replace(b'Wed, 21 Oct 2020 09:00:00 +0300', stamp))
            return response
    normal = poll_feed(factory, FEED, ['DEMO_A'], fetcher=RecentFetcher(count=1), force=True)
    random = poll_feed(factory, FEED, ['DEMO_A'], fetcher=RecentFetcher(count=2), force=True,
                       sample_seed='unpaid-test', max_articles=2)
    assert normal['downloaded'] == 1 and random['downloaded'] == 1
    while process_next_job(factory):
        pass
    with factory() as db:
        normal_source = db.get(RssDelivery, (normal['article_ids'][0], 'DEMO_A')).source_id
        assert db.get(RssSourcePolicy, normal_source) is None
        policies = list(db.scalars(select(RssSourcePolicy)))
        assert len(policies) == 1 and policies[0].suppress_automatic_ai is True
        suppressed = policies[0].source_id
        event_ids = set(db.scalars(select(Event.id).where(Event.source_id == suppressed)))
    monkeypatch.setenv('AI_AUTOMATION_ENABLED', '1')
    monkeypatch.setenv('AI_AUTOMATION_COMPANY_IDS', 'DEMO_A')
    monkeypatch.setattr('app.intelligence.article_cache._public_snapshot',
                        lambda *args: pytest.fail('Suppressed test sources must be skipped before AI dispatch'))
    report = dispatch_once(factory, settings, AutomationConfig.from_env(), event_ids=event_ids)
    assert report['enabled'] is True and report['scanned'] == 1
    assert report['extractions_queued'] == report['cards_queued'] == 0
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(AutomationRecord)) == 0
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0
        assert set(db.scalars(select(Job.operation))) == {'publication_ingest'}
