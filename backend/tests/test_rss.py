"""Real feed/article parsing, durable delivery and existing API/worker boundaries."""
from __future__ import annotations

import gzip
import io
import json

import pytest
from sqlalchemy import func, select

from app.news import Event, Job, SourceDocument
from app.rss import (FeedConfig, FeedState, RssArticle, RssDelivery, _claim, collect_once,
                     load_config, target_companies)
from app.rss_fetch import CollectionError, Download, PublicFetcher, public_addresses, public_url
from app.rss_parse import article_text, parse_feed
from app.worker import process_next_job
from test_news import app_store

HOSTS = frozenset({'news.example.org'})
FEED = FeedConfig(id='test-news', url='https://news.example.org/feed', article_hosts=list(HOSTS), poll_seconds=60)
RSS = b'''<?xml version="1.0"?><rss version="2.0"><channel><title>News</title><item>
<title>Economic decision for companies</title><link>https://news.example.org/article</link>
<pubDate>Wed, 21 Oct 2026 09:00:00 +0300</pubDate><description>Only a short feed teaser.</description>
</item></channel></rss>'''
ARTICLE = b'''<!doctype html><html><head><title>Economic decision for companies</title></head>
<body><nav>Navigation only: About Contact Login</nav><main><article><h1>Economic decision for companies</h1>
<p>The central bank announced its decision on interest rates. Companies need to check the currency,
remaining maturity and fixed or floating rate clauses of their actual financing contracts.</p>
<p>The announcement does not disclose debt balances, cash flow forecasts or company specific margins.
The reported decision alone cannot establish the size of a monetary gain or loss for any company.</p>
</article></main><footer>Subscribe to advertising and read comments</footer></body></html>'''


class FixtureFetcher:
    def __init__(self):
        self.calls = []
        self.feed_status = 200
        self.article_error = None
        self.feed_body = RSS

    def get(self, url, hosts, validators=None):
        assert hosts == HOSTS
        self.calls.append((url, validators))
        if url == FEED.url:
            return Download(url, self.feed_status, {'etag': '"revision-1"', 'content-type': 'application/rss+xml'},
                            self.feed_body if self.feed_status == 200 else b'')
        if self.article_error:
            raise CollectionError(self.article_error)
        assert url == 'https://news.example.org/article'
        return Download(url, 200, {'content-type': 'text/html; charset=utf-8'}, ARTICLE)


def test_rss_and_atom_preserve_dates_links_and_unknown_time():
    items = parse_feed(RSS, FEED.url)
    assert items[0].published_at == '2026-10-21T06:00:00+00:00'
    assert items[0].url == 'https://news.example.org/article'
    atom = b'''<feed xmlns="http://www.w3.org/2005/Atom" xml:base="https://news.example.org/">
      <entry><title>A new decision</title><link rel="self" href="entry.xml"/>
      <link rel="alternate" href="article"/><updated>2026-10-21T09:00:00Z</updated></entry>
      <entry><title>Another decision</title><link href="other"/>
      <published>2026-10-21T12:00:00+03:00</published></entry></feed>'''
    parsed = parse_feed(atom, FEED.url)
    assert parsed[0].url == 'https://news.example.org/article' and parsed[0].published_at is None
    assert parsed[1].published_at == '2026-10-21T09:00:00+00:00'
    assert parse_feed(RSS.replace(b'+0300', b''), FEED.url)[0].published_at is None


@pytest.mark.parametrize('body', [b'<html>an error page</html>', b'<rss>',
    b'<!DOCTYPE rss [<!ENTITY secret SYSTEM "file:///etc/passwd">]><rss>&secret;</rss>'])
def test_invalid_or_entity_bearing_feed_is_rejected(body):
    with pytest.raises(CollectionError):
        parse_feed(body, FEED.url)


def test_article_extraction_uses_body_not_teaser_or_navigation():
    text = article_text(ARTICLE, 'https://news.example.org/article', 'text/html; charset=utf-8')
    assert 'fixed or floating rate' in text and 'company specific margins' in text
    assert 'Only a short feed teaser' not in text
    assert 'Navigation only' not in text and 'Subscribe to advertising' not in text
    with pytest.raises(CollectionError, match='article_not_html'):
        article_text(b'%PDF data', FEED.url, 'application/pdf')
    with pytest.raises(CollectionError, match='article_text_missing'):
        article_text(b'<html><body>Login required</body></html>', FEED.url, 'text/html')


def test_collection_queue_existing_worker_api_and_company_isolation(app_store):
    client, factory, headers = app_store
    fetcher = FixtureFetcher()
    report = collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)[0]
    assert report['status'] == 'collected' and report['downloaded'] == report['queued'] == 1
    assert client.get('/api/v1/companies/U_A/events', headers=headers['U_A']).json()['items'] == []
    assert process_next_job(factory)
    event = client.get('/api/v1/companies/U_A/events', headers=headers['U_A']).json()['items'][0]
    source = client.get('/api/v1/sources/' + event['source_ids'][0], headers=headers['U_A']).json()
    assert source['kind'] == 'rss_article' and source['fetched_at'] and source['url_verified'] is True
    assert source['completeness'] == 'extracted_article_text'
    assert source['published_at'] == '2026-10-21T06:00:00+00:00'
    assert 'fixed or floating rate' in source['text']
    assert event['sources'][0]['url_verified'] is True and event['facts'] == []
    assert client.get('/api/v1/events/' + event['id'], headers=headers['U_B']).status_code == 404
    assert client.get('/api/v1/sources/' + source['id'], headers=headers['U_B']).status_code == 404
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 1
        assert db.scalar(select(Job.operation)) == 'publication_ingest'  # No paid analysis task.


def test_two_companies_share_download_but_keep_distinct_sources_and_events(app_store):
    client, factory, headers = app_store
    fetcher = FixtureFetcher()
    report = collect_once(factory, [FEED], ['U_A', 'U_B'], fetcher=fetcher, force=True)[0]
    assert report['downloaded'] == 1 and report['queued'] == 2 and len(fetcher.calls) == 2
    while process_next_job(factory):
        pass
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(RssArticle)) == 1
        assert db.scalar(select(func.count()).select_from(SourceDocument)) == 2
        assert db.scalar(select(func.count()).select_from(Event)) == 2
        assert db.scalar(select(func.count()).select_from(RssDelivery)) == 2
    own = client.get('/api/v1/companies/U_A/events', headers=headers['U_A']).json()['items'][0]
    other = client.get('/api/v1/companies/U_B/events', headers=headers['U_B']).json()['items'][0]
    assert own['id'] != other['id']
    assert client.get('/api/v1/events/' + own['id'], headers=headers['U_B']).status_code == 404


def test_restart_repeated_feed_and_304_do_not_duplicate_and_new_target_gets_cache(app_store):
    _, factory, _ = app_store
    fetcher = FixtureFetcher()
    collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)
    repeated = collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)[0]
    assert repeated['reused'] == 1 and repeated['queued'] == 0
    assert len([url for url, _ in fetcher.calls if url.endswith('/article')]) == 1
    fetcher.feed_status = 304
    report = collect_once(factory, [FEED], ['U_A', 'U_B'], fetcher=fetcher, force=True)[0]
    assert report['status'] == 'not_modified' and report['queued'] == 1
    assert fetcher.calls[-1][1]['If-None-Match'] == '"revision-1"'
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 2
    # 304 responses without new validator headers preserve previous ones.
    class NotModified(FixtureFetcher):
        def get(self, url, hosts, validators=None):
            assert validators['If-None-Match'] == '"revision-1"'
            return Download(url, 304, {}, b'')
    collect_once(factory, [FEED], ['U_A'], fetcher=NotModified(), force=True)
    with factory() as db:
        assert db.get(FeedState, FEED.id).etag == '"revision-1"'


def test_failed_article_is_retried_instead_of_hidden_by_etag(app_store):
    _, factory, _ = app_store
    fetcher = FixtureFetcher()
    fetcher.article_error = 'download_failed'
    report = collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)[0]
    assert report['status'] == 'partial' and report['queued'] == 0
    with factory() as db:
        assert db.get(FeedState, FEED.id).etag is None
        assert db.get(FeedState, FEED.id).last_error == 'download_failed'
        assert db.scalar(select(func.count()).select_from(SourceDocument)) == 0
    fetcher.article_error = None
    assert collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)[0]['queued'] == 1
    assert fetcher.calls[-2][1]['If-None-Match'] is None


def test_persistent_schedule_and_active_lease_survive_restart(app_store):
    _, factory, _ = app_store
    fetcher = FixtureFetcher()
    first = collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, clock=lambda: 1_800_000_000)[0]
    assert first['status'] == 'collected'
    assert collect_once(factory, [FEED], ['U_A'], fetcher=fetcher,
                        clock=lambda: 1_800_000_001)[0]['status'] == 'not_due'
    assert len(fetcher.calls) == 2
    assert _claim(factory, FEED, 1_800_000_100, False)
    assert _claim(factory, FEED, 1_800_000_101, True) is None
    assert _claim(factory, FEED, 1_800_001_000, False)


def test_out_of_allowlist_links_are_rejected_without_download(app_store):
    _, factory, _ = app_store
    fetcher = FixtureFetcher()
    fetcher.feed_body = RSS.replace(b'https://news.example.org/article', b'http://169.254.169.254/latest')
    report = collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)[0]
    assert report['rejected'] == 1 and report['queued'] == 0 and len(fetcher.calls) == 1


def test_targets_are_bounded_and_validated_before_network(app_store, monkeypatch):
    _, factory, _ = app_store
    fetcher = FixtureFetcher()
    with pytest.raises(CollectionError, match='rss_targets_not_allowed'):
        collect_once(factory, [FEED], ['FOREIGN'], fetcher=fetcher)
    monkeypatch.setenv('DEMO_COMPANY_IDS', 'U_A,U_B,NOT_IMPORTED')
    report = collect_once(factory, [FEED], ['NOT_IMPORTED'], fetcher=fetcher)
    assert report[0]['status'] == 'waiting_for_company_import' and fetcher.calls == []
    monkeypatch.setenv('DEMO_COMPANY_IDS', ','.join(f'C{i}' for i in range(21)))
    with pytest.raises(CollectionError, match='rss_targets_not_allowed'):
        target_companies()


@pytest.mark.parametrize('url', ['http://127.0.0.1/a', 'http://169.254.169.254/a',
    'https://user:secret@news.example.org/a', 'https://news.example.org:8443/a',
    'https://evil.example.org/a', 'file:///etc/passwd', 'https://news.example.org/a\r\nx-header: bad'])
def test_fetcher_rejects_invalid_urls_before_dns(url, monkeypatch):
    monkeypatch.setattr('socket.getaddrinfo', lambda *a, **k: pytest.fail('No DNS for rejected URL'))
    with pytest.raises(CollectionError, match='url_not_allowed'):
        PublicFetcher().get(url, HOSTS)


@pytest.mark.parametrize('addresses', [['127.0.0.1'], ['169.254.169.254'], ['::1'],
    ['93.184.216.34', '10.0.0.1'], ['::ffff:127.0.0.1']])
def test_all_dns_answers_must_be_public(addresses, monkeypatch):
    monkeypatch.setattr('socket.getaddrinfo', lambda *a, **k: [(2, 1, 6, '', (ip, 443)) for ip in addresses])
    with pytest.raises(CollectionError, match='private_address_blocked'):
        public_addresses('news.example.org', 443)


def test_redirect_target_is_checked_before_another_request(monkeypatch):
    seen = []
    def request(url, headers, deadline):
        seen.append(url)
        return Download(url, 302, {'location': 'http://169.254.169.254/'}, b'')
    monkeypatch.setattr('app.rss_fetch._request', request)
    with pytest.raises(CollectionError, match='url_not_allowed'):
        PublicFetcher().get(FEED.url, HOSTS)
    assert seen == [FEED.url]


def test_default_config_loads_and_duplicate_feed_ids_fail(tmp_path):
    defaults = {feed.id: feed for feed in load_config()}
    assert set(defaults) == {'cbr-press', 'cbr-events', 'interfax-business', 'stroygaz',
                            'agroinvestor', 'mintrans', 'eec', 'fns', 'trendforce', 'russoft'}
    assert defaults['interfax-business'].article_path_prefixes == ['/business/']
    path = tmp_path / 'rss.json'
    path.write_text(json.dumps({'version': 1, 'feeds': [FEED.model_dump(), FEED.model_dump()]}))
    with pytest.raises(CollectionError, match='invalid_rss_config'):
        load_config(path)


def test_russoft_relative_listing_dates_use_only_explicit_article_publication_metadata():
    from app.rss_parse import discover_items, article_publication_date
    body = ('<div><a class="post__name" href="/news/new-software-market/">Новости рынка программного обеспечения</a>'
            '<div class="post__date">Сегодня 15:41</div></div>').encode()
    item = discover_items(body, 'https://russoft.org/news/', 'russoft_html')[0]
    assert item.published_at is None
    article = b'<html><head><meta property="article:published_time" content="2026-10-21T15:41:00+03:00"></head></html>'
    assert article_publication_date(article) == '2026-10-21T12:41:00+00:00'
    assert article_publication_date(article.replace(b'published_time', b'modified_time')) is None
    assert article_publication_date(b'<html>Today 15:41</html>') is None


def test_article_shared_by_two_feeds_is_available_for_304_catchup(app_store):
    _, factory, _ = app_store
    first = FixtureFetcher()
    collect_once(factory, [FEED], ['U_A'], fetcher=first, force=True)
    second_feed = FEED.model_copy(update={'id': 'second-feed'})
    second = FixtureFetcher()
    report = collect_once(factory, [second_feed], ['U_A'], fetcher=second, force=True)[0]
    assert report['reused'] == 1 and report['downloaded'] == 0
    second.feed_status = 304
    report = collect_once(factory, [second_feed], ['U_B'], fetcher=second, force=True)[0]
    assert report['queued'] == 1 and report['status'] == 'not_modified'
    assert len(second.calls) == 2  # Only the two feed GETs; article reused across feeds.


def test_stale_collector_cannot_publish_or_clear_newer_lease(app_store):
    _, factory, _ = app_store
    class StaleFetcher(FixtureFetcher):
        def get(self, url, hosts, validators=None):
            response = super().get(url, hosts, validators)
            if url.endswith('/article'):
                with factory.begin() as db:
                    state = db.get(FeedState, FEED.id)
                    state.lease_token = 'a-newer-collector'
            return response
    report = collect_once(factory, [FEED], ['U_A'], fetcher=StaleFetcher(), force=True)[0]
    assert report['status'] == 'failed' and report['errors'] == ['rss_lease_lost']
    with factory() as db:
        assert db.get(FeedState, FEED.id).lease_token == 'a-newer-collector'
        assert db.scalar(select(func.count()).select_from(SourceDocument)) == 0
        assert db.scalar(select(func.count()).select_from(Job)) == 0


def test_https_redirect_cannot_downgrade_or_keep_conditionals(monkeypatch):
    seen = []
    def request(url, headers, deadline):
        seen.append((url, headers))
        if len(seen) == 1:
            return Download(url, 302, {'location': '/new-feed'}, b'')
        assert 'If-None-Match' not in headers
        return Download(url, 200, {}, RSS)
    monkeypatch.setattr('app.rss_fetch._request', request)
    result = PublicFetcher().get(FEED.url, HOSTS, {'If-None-Match': 'old-etag'})
    assert result.url == 'https://news.example.org/new-feed'
    monkeypatch.setattr('app.rss_fetch._request', lambda url, headers, deadline:
        Download(url, 302, {'location': 'http://news.example.org/feed'}, b''))
    with pytest.raises(CollectionError, match='https_downgrade_blocked'):
        PublicFetcher().get(FEED.url, HOSTS)


def test_wire_download_pins_public_ip_uses_original_sni_and_bounds_gzip(monkeypatch):
    import app.rss_fetch as network
    calls = []
    class Wire:
        def close(self): pass
        def settimeout(self, value): pass
    class Context:
        def set_alpn_protocols(self, protocols): pass
        def wrap_socket(self, wire, *, server_hostname):
            calls.append(('sni', server_hostname))
            return wire
    body = gzip.compress(b'x' * 300)
    class Response:
        status = 200
        def __init__(self): self.stream = io.BytesIO(body)
        def getheaders(self): return [('Content-Length', str(len(body))), ('Content-Encoding', 'gzip')]
        def read1(self, count): return self.stream.read(count)
    class Connection:
        def __init__(self, host, port, *, timeout):
            calls.append(('host', host, port))
            self.timeout = timeout
        def request(self, method, target, *, headers): calls.append(('get', method, target))
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(network, 'public_addresses', lambda host, port: ['93.184.216.34'])
    monkeypatch.setattr(network.socket, 'create_connection', lambda address, **kw: calls.append(('connect', address)) or Wire())
    monkeypatch.setattr(network.ssl, 'create_default_context', Context)
    monkeypatch.setattr(network.http.client, 'HTTPConnection', Connection)
    assert PublicFetcher().get(FEED.url, HOSTS).body == b'x' * 300
    assert ('connect', ('93.184.216.34', 443)) in calls
    assert ('sni', 'news.example.org') in calls and ('host', 'news.example.org', 443) in calls
    monkeypatch.setattr(network, 'MAX_BYTES', 100)
    with pytest.raises(CollectionError, match='response_too_large'):
        PublicFetcher().get(FEED.url, HOSTS)


def feed_entries(paths):
    from xml.sax.saxutils import escape
    entries = ''.join('<item><title>Economic news for companies</title><link>'
                      + escape('https://news.example.org' + path) + '</link></item>' for path in paths)
    return ('<rss version="2.0"><channel>' + entries + '</channel></rss>').encode()


class RoutingFetcher(FixtureFetcher):
    def __init__(self, paths, *, final_path=None):
        super().__init__()
        self.feed_body = feed_entries(paths)
        self.final_path = final_path

    def get(self, url, hosts, validators=None):
        if url == FEED.url:
            return super().get(url, hosts, validators)
        self.calls.append((url, validators))
        final = 'https://news.example.org' + self.final_path if self.final_path else url
        return Download(final, 200, {'content-type': 'text/html'}, ARTICLE)


def test_business_filter_runs_before_download_limit(app_store):
    _, factory, _ = app_store
    feed = FEED.model_copy(update={'max_items': 1, 'article_path_prefixes': ['/business/']})
    fetcher = RoutingFetcher(['/world/one', '/politics/two', '/business/accepted', '/business/later'])
    result = collect_once(factory, [feed], ['U_A'], fetcher=fetcher, force=True)[0]
    assert result['status'] == 'collected' and result['queued'] == result['downloaded'] == 1
    assert result['feed_items'] == 4 and result['filtered'] == 2
    assert [url for url, _ in fetcher.calls] == [FEED.url, 'https://news.example.org/business/accepted']


def test_no_business_items_means_no_article_download_or_job(app_store):
    _, factory, _ = app_store
    feed = FEED.model_copy(update={'article_path_prefixes': ['/business/']})
    fetcher = RoutingFetcher(['/world/one', '/politics/two', '/business-like/three'])
    result = collect_once(factory, [feed], ['U_A'], fetcher=fetcher, force=True)[0]
    assert result['status'] == 'collected' and result['filtered'] == 3 and result['queued'] == 0
    assert len(fetcher.calls) == 1
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 0


def test_business_link_redirected_to_another_section_is_not_published(app_store):
    _, factory, _ = app_store
    feed = FEED.model_copy(update={'article_path_prefixes': ['/business/']})
    fetcher = RoutingFetcher(['/business/article'], final_path='/world/article')
    result = collect_once(factory, [feed], ['U_A'], fetcher=fetcher, force=True)[0]
    assert result['status'] == 'collected' and result['filtered'] == 1 and result['queued'] == 0
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(RssArticle)) == 0


def test_filter_change_invalidates_etag_and_304_rechecks_cached_articles(app_store):
    _, factory, _ = app_store
    fetcher = RoutingFetcher(['/world/one', '/business/two'])
    collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)
    filtered = FEED.model_copy(update={'article_path_prefixes': ['/business/']})
    result = collect_once(factory, [filtered], ['U_A'], fetcher=fetcher, force=True)[0]
    assert result['filtered'] == 1 and result['queued'] == 0
    assert fetcher.calls[-1][1] == {}  # Filter policy changed; old validators cannot hide new candidates.
    fetcher.feed_status = 304
    result = collect_once(factory, [filtered], ['U_B'], fetcher=fetcher, force=True)[0]
    assert result['status'] == 'not_modified' and result['queued'] == 1 and result['filtered'] == 1
    with factory() as db:
        copied = db.scalars(select(SourceDocument).where(SourceDocument.company_id == 'U_B')).all()
        assert len(copied) == 1 and '/business/' in copied[0].source_url


def test_cached_redirect_is_rechecked_after_filter_change(app_store):
    _, factory, _ = app_store
    fetcher = RoutingFetcher(['/business/article'], final_path='/world/article')
    collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)
    filtered = FEED.model_copy(update={'article_path_prefixes': ['/business/']})
    result = collect_once(factory, [filtered], ['U_B'], fetcher=fetcher, force=True)[0]
    assert result['filtered'] == 1 and result['queued'] == 0
    assert len(fetcher.calls) == 3  # Article isn't downloaded twice merely to recheck its final URL.


@pytest.mark.parametrize('prefix', ['business/', '//business/', '/business', '/business/?x=', '/../business/', '/business/%2f/'])
def test_invalid_path_filter_configuration_is_rejected(prefix):
    with pytest.raises(ValueError):
        FeedConfig.model_validate({**FEED.model_dump(), 'article_path_prefixes': [prefix]})


def test_all_configured_sources_enter_existing_queue_and_deduplicate(app_store):
    from urllib.parse import urlsplit
    _, factory, _ = app_store
    feeds = load_config()
    feed_by_url = {feed.url: feed for feed in feeds}
    class ConfiguredSources:
        def __init__(self): self.article_downloads = 0
        def get(self, url, hosts, validators=None):
            if url in feed_by_url:
                feed = feed_by_url[url]
                path = {
                    'interfax-business': '/business/story', 'mintrans': '/press-center/news/12345',
                    'eec': '/news/new-sector-rule/', 'fns': '/rn77/news/activities_fts/12345/',
                    'trendforce': '/presscenter/news/20261020-12345.html',
                    'russoft': '/news/new-sector-rule/',
                }.get(feed.id, '/articles/' + feed.id)
                item_url = 'https://' + urlsplit(feed.url).hostname + path
                body = ('<rss><channel><item><title>Economic news for companies</title><link>'
                        + item_url + '</link></item></channel></rss>').encode()
                if feed.format == 'mintrans_html':
                    body = ('<div class="news-inf"><div><span class="date-span">20 октября 2026</span></div>'
                            f'<a class="news-text" href="{path}">Economic news for companies</a></div>').encode()
                elif feed.format == 'eec_html':
                    body = (f'<a class="NewsItem" href="{path}"><div class="NewsItem__Date">20 октября 2026</div>'
                            '<div class="NewsItem__Title">Economic news for companies</div></a>').encode()
                elif feed.format == 'trendforce_html':
                    body = (f'<div><h3><a class="title-link" href="{path}">Economic news for companies</a></h3>'
                            '<h4>20 October 2026</h4></div>').encode()
                elif feed.format == 'russoft_html':
                    body = (f'<div><a class="post__name" href="{path}">Economic news for companies</a>'
                            '<div class="post__date">20 октября 2026</div></div>').encode()
                return Download(url, 200, {'content-type': 'application/rss+xml'}, body)
            assert urlsplit(url).hostname in hosts
            self.article_downloads += 1
            return Download(url, 200, {'content-type': 'text/html'}, ARTICLE)
    fetcher = ConfiguredSources()
    reports = collect_once(factory, feeds, ['U_A', 'U_B'], fetcher=fetcher, force=True)
    assert all(row['status'] == 'collected' and row['queued'] == 2 for row in reports)
    while process_next_job(factory):
        pass
    reports = collect_once(factory, feeds, ['U_A', 'U_B'], fetcher=fetcher, force=True)
    assert all(row['queued'] == 0 for row in reports)
    assert fetcher.article_downloads == len(feeds)
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Event)) == 2 * len(feeds)
        assert db.scalar(select(func.count()).select_from(Job)) == 2 * len(feeds)
        assert db.scalar(select(func.count()).select_from(Job).where(Job.operation != 'publication_ingest')) == 0


def test_existing_feed_database_keeps_state_when_selection_table_is_added(app_store):
    from app.core import Base
    _, factory, _ = app_store
    engine = factory.kw['bind']
    with factory.begin() as db:
        db.add(FeedState(id=FEED.id, url=FEED.url, etag='"old-config"', next_poll_at=0,
                         lease_until=0, failures=0))
    # Simulate the first collector version, which already has its durable feed state.
    with engine.begin() as connection:
        connection.exec_driver_sql('DROP TABLE rss_feed_selections')
    Base.metadata.create_all(engine)
    with factory() as db:
        assert db.get(FeedState, FEED.id).etag == '"old-config"'
    fetcher = FixtureFetcher()
    result = collect_once(factory, [FEED], ['U_A'], fetcher=fetcher, force=True)[0]
    assert result['status'] == 'collected' and result['queued'] == 1
    assert fetcher.calls[0][1] == {}  # Old unknown selection must not reuse its ETag.
