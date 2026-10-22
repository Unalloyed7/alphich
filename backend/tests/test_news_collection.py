"""Unpaid click-to-news collection: durable state, isolation and novel samples."""
from datetime import datetime, timezone
import time

from sqlalchemy import func, select

from app.news import Event, Job, SourceDocument
from app.news_collection import NewsCollectionRun, collection_tick
from app.rss import FeedConfig, FeedState, RssArticle, RssDelivery, poll_feed
from app.rss_fetch import Download
from app.worker import process_next_job
from test_news import app_store


FEED = FeedConfig(id='sample-fixture', url='https://news.example.org/feed',
                  article_hosts=['news.example.org'], poll_seconds=300, max_items=10)


class SampleFetcher:
    def __init__(self, count=8, future=False):
        self.count, self.future, self.calls = count, future, []

    def get(self, url, hosts, validators=None):
        self.calls.append(url)
        if url == FEED.url:
            date = 'Wed, 21 Oct 2037 09:00:00 +0300' if self.future else 'Wed, 21 Oct 2020 09:00:00 +0300'
            items = ''.join(f'<item><title>Economic business announcement {i}</title>'
                f'<link>https://news.example.org/article-{i}</link><pubDate>{date}</pubDate></item>'
                for i in range(self.count))
            body = ('<rss version="2.0"><channel>' + items + '</channel></rss>').encode()
            return Download(url, 200, {'content-type': 'application/rss+xml', 'etag': 'sample-v1'}, body)
        body = ('<html><main><article><h1>Business announcement</h1><p>'
            'The authority published a verified announcement for businesses. '
            'Companies should inspect the exact eligibility, region and actual scale before taking action. '
            'The source does not disclose individual company exposure or contract obligations. '
            'This is the full article text rather than a feed teaser. '
            + url + '</p></article></main></html>').encode()
        return Download(url, 200, {'content-type': 'text/html; charset=utf-8'}, body)


def start(store, monkeypatch, count=3, key='news-sample-first-001'):
    monkeypatch.setattr('app.rss.load_config', lambda *args, **kwargs: [FEED])
    client, _, headers = store
    return client.post('/api/v1/companies/U_A/news-search', json={'count': count},
        headers={**headers['U_A'], 'Idempotency-Key': key})


def detail(store, run_id):
    client, _, headers = store
    return client.get('/api/v1/news-search/' + run_id, headers=headers['U_A']).json()


def advance(store, fetcher):
    _, factory, _ = store
    def poll(factory, feed, targets, **kwargs):
        return poll_feed(factory, feed, targets, fetcher=fetcher, **kwargs)
    assert collection_tick(factory, feeds=[FEED], poll=poll)
    while process_next_job(factory):
        pass
    collection_tick(factory, feeds=[FEED], poll=poll)


def test_start_requires_auth_own_company_valid_count_and_idempotency(app_store, monkeypatch):
    client, factory, headers = app_store
    path = '/api/v1/companies/U_A/news-search'
    assert client.post(path, json={'count': 2}).status_code == 401
    assert client.post(path, json={'count': 2}, headers=headers['U_B']).status_code == 403
    assert client.post(path, json={'count': 2}, headers=headers['U_A']).status_code == 400
    for bad in (0, 101, True, 1.5):
        assert client.post(path, json={'count': bad}, headers=headers['U_A']).status_code == 422
    first = start(app_store, monkeypatch)
    assert first.status_code == 202, first.text
    assert first.json()['requested_count'] == 3 and first.json()['status'] == 'queued'
    assert start(app_store, monkeypatch).json()['id'] == first.json()['id']
    assert start(app_store, monkeypatch, count=4).status_code == 409
    assert client.get('/api/v1/news-search/' + first.json()['id'], headers=headers['U_B']).status_code == 404
    another = client.post('/api/v1/demo-session', json={'company_id': 'U_A', 'role': 'ceo'}).json()['token']
    other_headers = {'X-Demo-Session': another, 'Idempotency-Key': 'other-owner-first'}
    assert client.get('/api/v1/news-search/' + first.json()['id'], headers=other_headers).status_code == 404
    assert client.post(path, json={'count': 2}, headers=other_headers).status_code == 409
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(NewsCollectionRun)) == 1
        assert db.scalar(select(func.count()).select_from(Job)) == 0


def test_collects_requested_random_articles_through_real_ingestion_without_llm(app_store, monkeypatch):
    _, factory, _ = app_store
    first = start(app_store, monkeypatch, count=3).json()
    fetcher = SampleFetcher()
    advance(app_store, fetcher)
    result = detail(app_store, first['id'])
    assert result['status'] == 'succeeded' and result['stop_reason'] == 'target_reached', result
    assert result['collected_count'] == 3 and result['checked_sources'] == 1
    assert len({item['article_id'] for item in result['items']}) == 3
    assert all(item['event_id'] for item in result['items'])
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Event)) == 3
        assert set(db.scalars(select(Job.operation))) == {'publication_ingest'}
        assert set(db.scalars(select(SourceDocument.kind))) == {'rss_article'}


def test_next_run_uses_new_articles_and_reuses_background_cache(app_store, monkeypatch):
    _, factory, _ = app_store
    fetcher = SampleFetcher()
    first = start(app_store, monkeypatch, count=3).json()
    advance(app_store, fetcher)
    old = {item['article_id'] for item in detail(app_store, first['id'])['items']}
    with factory.begin() as db:
        db.get(NewsCollectionRun, first['id']).created_at -= 61
        db.get(FeedState, FEED.id).last_checked_at -= 61
    second = start(app_store, monkeypatch, count=3, key='news-sample-next-002')
    assert second.status_code == 202, second.text
    advance(app_store, fetcher)
    current = detail(app_store, second.json()['id'])
    assert current['status'] == 'succeeded', current
    assert len(old | {item['article_id'] for item in current['items']}) == 6
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(RssArticle)) == 6
        assert db.scalar(select(func.count()).select_from(RssDelivery)) == 6


def test_cached_background_articles_can_be_sampled_without_duplicate_jobs(app_store, monkeypatch):
    _, factory, _ = app_store
    fetcher = SampleFetcher(count=4)
    poll_feed(factory, FEED, ['U_A'], fetcher=fetcher, force=True)
    while process_next_job(factory):
        pass
    ident = start(app_store, monkeypatch, count=3).json()['id']
    advance(app_store, fetcher)
    result = detail(app_store, ident)
    assert result['status'] == 'succeeded' and result['reused'] == 3, result
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 4


def test_exhaustion_returns_partial_and_cooldown_and_stop_are_explicit(app_store, monkeypatch):
    client, factory, headers = app_store
    ident = start(app_store, monkeypatch, count=10).json()['id']
    advance(app_store, SampleFetcher(count=2))
    result = detail(app_store, ident)
    assert result['status'] == 'partial' and result['stop_reason'] == 'sources_exhausted'
    assert result['collected_count'] == 2
    assert start(app_store, monkeypatch, key='new-start-too-soon').status_code == 429
    with factory.begin() as db:
        db.get(NewsCollectionRun, ident).created_at -= 61
    next_id = start(app_store, monkeypatch, key='new-start-cancelled').json()['id']
    stopped = client.post('/api/v1/news-search/' + next_id + '/stop', headers=headers['U_A'])
    assert stopped.json()['status'] == 'cancelled'
    assert not collection_tick(factory, feeds=[FEED], poll=lambda *a, **kw: (_ for _ in ()).throw(AssertionError()))


def test_future_publication_is_not_returned_and_deadline_recovers_abandoned_run(app_store, monkeypatch):
    _, factory, _ = app_store
    ident = start(app_store, monkeypatch, count=2).json()['id']
    advance(app_store, SampleFetcher(count=2, future=True))
    result = detail(app_store, ident)
    assert result['items'] == [] and result['status'] == 'partial', result
    with factory.begin() as db:
        run = db.get(NewsCollectionRun, ident)
        run.status, run.active_key = 'collecting', 'U_A'
        run.deadline_at = int(time.time()) - 1
        run.lease_until = int(time.time()) - 1
    assert collection_tick(factory, feeds=[FEED])
    result = detail(app_store, ident)
    assert result['status'] == 'failed' and result['stop_reason'] == 'time_limit'
