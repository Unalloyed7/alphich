"""Official listing layouts use the existing bounded public fetch/ingestion path."""
import pytest
from sqlalchemy import func, select

from app.news import Job, SourceDocument
from app.rss import FeedConfig, collect_once
from app.rss_fetch import CollectionError, Download
from app.rss_parse import discover_items
from test_news import app_store
from test_rss import ARTICLE, FEED, FixtureFetcher


@pytest.mark.parametrize("format,body,path,when", [
    ("mintrans_html", '<div class="news-inf"><div><span class="date-span dark">20 Октября 2026</span></div>'
     '<a class="news-text fc-blue" href="/press-center/news/13058">Новое правило перевозок</a></div>',
     "/press-center/news/13058", "2026-10-20T20:59:59+00:00"),
    ("eec_html", '<a class="NewsItem" href="/news/customs-rules/"><div class="NewsItem__Date">20 октября 2026</div>'
     '<div class="NewsItem__Title">Изменение таможенных правил</div></a>',
     "/news/customs-rules/", "2026-10-20T20:59:59+00:00"),
    ("trendforce_html", '<div><h3><a class="title-link" href="/presscenter/news/20261020-13270.html">Memory market outlook</a></h3>'
     '<h4><strong>20 October 2026</strong></h4></div>',
     "/presscenter/news/20261020-13270.html", "2026-10-20T15:59:59+00:00"),
])
def test_inspected_layout_extracts_title_link_and_conservative_publisher_day(format, body, path, when):
    links = '<nav><a href="/login/">Войти</a><a href="/news/">Новости</a></nav>'
    items = discover_items((links + body + body).encode(), FEED.url, format)
    assert len(items) == 1
    assert items[0].url == "https://news.example.org" + path
    assert items[0].published_at == when
    assert "2026" not in items[0].title


def test_unknown_listing_day_never_uses_download_time_or_url_date():
    body = b'<div><h3><a class="title-link" href="/presscenter/news/20261020-13270.html">Memory market outlook</a></h3></div>'
    assert discover_items(body, FEED.url, "trendforce_html")[0].published_at is None


@pytest.mark.parametrize("body", [b'', b'<html>Captcha login page</html>', b'<a href="/press-center/news/123">Wrong layout</a>'])
def test_broken_or_protected_listing_fails_visibly_instead_of_becoming_empty_success(body):
    with pytest.raises(CollectionError):
        discover_items(body, FEED.url, "mintrans_html")


def test_malicious_listing_url_is_blocked_before_download_and_creates_no_job(app_store):
    _, factory, _ = app_store
    feed = FEED.model_copy(update={"format": "eec_html"})
    class Malicious(FixtureFetcher):
        def get(self, url, hosts, validators=None):
            assert url == FEED.url  # No request to the discovered internal URL.
            self.calls.append(url)
            body = b'<a class="NewsItem" href="http://127.0.0.1/news/secret/"><div class="NewsItem__Title">Injected private target</div></a>'
            return Download(url, 200, {"content-type": "text/html"}, body)
    fetcher = Malicious()
    result = collect_once(factory, [feed], ["U_A"], fetcher=fetcher, force=True)[0]
    assert result["rejected"] == 1 and result["queued"] == 0 and len(fetcher.calls) == 1
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 0


def test_official_html_item_keeps_full_source_provenance_and_does_not_start_llm(app_store):
    _, factory, _ = app_store
    feed = FEED.model_copy(update={"format": "mintrans_html"})
    class Official(FixtureFetcher):
        def get(self, url, hosts, validators=None):
            self.calls.append(url)
            if url == FEED.url:
                body = ('<div class="news-inf"><div><span class="date-span">20 октября 2026</span></div>'
                    '<a class="news-text" href="/press-center/news/123">Новое требование перевозчикам</a></div>').encode()
                return Download(url, 200, {"content-type": "text/html"}, body)
            assert url == "https://news.example.org/press-center/news/123"
            return Download(url, 200, {"content-type": "text/html"}, ARTICLE)
    report = collect_once(factory, [feed], ["U_A"], fetcher=Official(), force=True)[0]
    assert report["status"] == "collected" and report["queued"] == 1
    with factory() as db:
        source = db.scalar(select(SourceDocument))
        assert source.source_url.endswith("/press-center/news/123")
        assert source.published_at == "2026-10-20T20:59:59+00:00"
        assert "fixed or floating rate" in source.body
        assert db.scalar(select(Job.operation)) == "publication_ingest"


def test_discovery_format_change_invalidates_cached_feed_policy():
    changed = FEED.model_copy(update={"format": "eec_html"})
    assert changed.selection_hash != FEED.selection_hash
    with pytest.raises(ValueError):
        FeedConfig.model_validate({**FEED.model_dump(), "format": "untrusted_xpath"})
