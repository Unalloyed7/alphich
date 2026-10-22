"""Bounded collector tests use fake public fetchers and local SQLite only."""
import gzip
import json
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from app.rss_fetch import CollectionError, Download
from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.corpus import collect, iter_articles, corpus_summary, load_sources, _discover


class FakeFetcher:
    def __init__(self, values):
        self.values, self.calls = values, []

    def get(self, url, hosts):
        self.calls.append(url)
        value = self.values[url]
        if isinstance(value, Exception):
            raise value
        body, content_type = value
        return Download(url, 200, {"content-type": content_type}, body)


class Clock:
    def __init__(self):
        self.value = 1_792_670_400  # 2026-10-22 UTC; independent of host timezone.

    def __call__(self):
        self.value += 2
        return self.value


def article(title, *, date=True, suffix="", published_at="2026-10-20T10:00:00+03:00"):
    body = ("Предприятия обсуждают развитие рынка, изменение спроса и производственных затрат. " + suffix + " ") * 10
    meta = f'<meta property="article:published_time" content="{published_at}">' if date else ""
    return ((f"<html><head>{meta}<meta property='og:title' content='{title}'></head>"
             f"<body><article><h1>{title}</h1><p>{body}</p></article></body></html>").encode(), "text/html")


def source(host="news.example.test", sid="one"):
    return {"id": sid, "name": sid, "allowed_hosts": [host], "article_path_prefixes": ["/news/"],
            "feeds": [{"url": f"https://{host}/rss", "kind": "rss"}], "status": "available",
            "quality_tier": "industry_editorial", "topics": ["trade"], "genres": ["news"]}


def rss(host, names):
    return (("<rss><channel>" + "".join(f"<item><title>Новость {name}</title><link>https://{host}/news/{name}</link></item>" for name in names)
             + "</channel></rss>").encode(), "application/rss+xml")


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.manifest = self.root / "sources.json"
        self.database = self.root / "corpus.sqlite"
        self.clock = Clock()

    def tearDown(self):
        self.temp.cleanup()

    def write_sources(self, values):
        self.manifest.write_text(json.dumps({"schema_version": "alphich-relevance-sources-v1", "sources": values}))

    def run_collect(self, fetcher, **kwargs):
        return collect(self.manifest, self.database, fetcher=fetcher, clock=self.clock, **kwargs)

    def test_rate_limit_persists_across_resume_then_retries_pending_url(self):
        host = "news.example.test"
        self.write_sources([source(host)])
        limited = [True]
        base = FakeFetcher({f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                            f"https://{host}/rss": rss(host, ["a", "b"]),
                            f"https://{host}/news/a": article("Первая публикация", suffix="Альфа"),
                            f"https://{host}/news/b": article("Вторая публикация", suffix="Бета")})
        class RateLimited:
            def get(inner, url, hosts):
                if limited[0] and "/news/" in url:
                    base.calls.append(url)
                    return Download(url, 429, {"retry-after": "7200"}, b"")
                return base.get(url, hosts)
        first = self.run_collect(RateLimited())
        self.assertEqual(first["stop_reason"], "publisher_cooldown")
        self.assertEqual(first["pending"], 2)
        self.assertEqual(first["requests"], 3)
        before = len(base.calls)
        second = self.run_collect(RateLimited())
        self.assertEqual(second["requests"], 0)
        self.assertEqual(len(base.calls), before)
        self.clock.value += 7201
        limited[0] = False
        self.assertEqual(self.run_collect(RateLimited())["new"], 2)

    def test_robots_rate_limit_preserves_seed_and_http_date_deadline(self):
        host = "news.example.test"
        self.write_sources([source(host)])
        from datetime import datetime, timezone
        from email.utils import format_datetime
        retry = self.clock.value + 10800
        class RateLimited:
            def get(inner, url, hosts):
                return Download(url, 503, {"retry-after": format_datetime(datetime.fromtimestamp(retry, timezone.utc), usegmt=True)}, b"")
        result = self.run_collect(RateLimited())
        self.assertEqual(result["stop_reason"], "publisher_cooldown")
        self.assertEqual(result["requests"], 1)
        self.assertEqual(result["pending"], 1)
        with sqlite3.connect(self.database) as db:
            self.assertAlmostEqual(db.execute("SELECT retry_at FROM publisher_holds").fetchone()[0], retry, delta=2)
        self.assertEqual(self.run_collect(RateLimited())["requests"], 0)

    def test_repeated_redirect_failure_stops_publisher_instead_of_draining_queue(self):
        host = "news.example.test"
        self.write_sources([source(host)])
        urls = [str(i) for i in range(20)]
        base = FakeFetcher({f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                            f"https://{host}/rss": rss(host, urls),
                            **{f"https://{host}/news/{i}": CollectionError("redirect_limit") for i in urls}})
        result = self.run_collect(base)
        self.assertEqual(result["errors"], 5)
        self.assertEqual(result["pending"], 15)
        self.assertEqual(result["stop_reason"], "publisher_cooldown")
        self.assertEqual(self.run_collect(base)["requests"], 0)

    def test_outside_window_quarantine_retains_publication_evidence(self):
        host = "news.example.test"
        self.write_sources([source(host)])
        base = FakeFetcher({f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                            f"https://{host}/rss": rss(host, ["a"]),
                            f"https://{host}/news/a": article("Архивная публикация", published_at="2025-10-20T10:00:00+03:00")})
        result = self.run_collect(base, published_after="2026-09-07")
        self.assertEqual(result["new"], 0)
        with sqlite3.connect(self.database) as db:
            row = db.execute("SELECT published_at,publication_evidence FROM quarantine").fetchone()
        self.assertTrue(row[0].startswith("2025-10-20"))
        self.assertEqual(json.loads(row[1])[0]["selector"], "meta.article:published_time")

    def test_request_bound_resume_dedup_and_missing_date_quarantine(self):
        host = "news.example.test"
        self.write_sources([source(host)])
        values = {f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                  f"https://{host}/rss": rss(host, ["a", "b", "c", "d"]),
                  f"https://{host}/news/a": article("Первая новость", suffix="Альфа"),
                  f"https://{host}/news/b": article("Первая новость", suffix="Альфа"),
                  f"https://{host}/news/c": article("Без даты публикации", date=False),
                  f"https://{host}/news/d": article("Другой материал", suffix="Бета")}
        fetcher = FakeFetcher(values)
        first = self.run_collect(fetcher, max_requests=3, max_articles=10)
        self.assertEqual(first["requests"], 3)
        self.assertEqual(first["new"], 1)
        self.assertEqual(first["pending"], 3)
        second = self.run_collect(fetcher, max_requests=10, max_articles=10)
        self.assertEqual(second["new"], 1)
        self.assertEqual(second["duplicates"], 1)
        self.assertEqual(second["quarantined"], 1)
        third = self.run_collect(fetcher, max_requests=10)
        self.assertEqual(third["requests"], 0)
        self.assertTrue(all(count == 1 for count in Counter(fetcher.calls).values()))
        articles = list(iter_articles(self.database))
        self.assertEqual(len(articles), 2)
        self.assertTrue(all(row["published_at"] != row["fetched_at"] for row in articles))
        self.assertEqual(corpus_summary(self.database)["queue"]["quarantine"], 1)
        self.assertEqual(self.database.stat().st_mode & 0o777, 0o600)

    def test_nested_gzip_sitemaps_resume_and_external_urls_never_fetched(self):
        host = "news.example.test"
        config = source(host)
        config["feeds"] = [{"url": f"https://{host}/sitemap-index.xml", "kind": "sitemap"}]
        self.write_sources([config])
        index = (f"<sitemapindex><sitemap><loc>https://{host}/sitemap-news.xml</loc></sitemap>"
                 "<sitemap><loc>https://evil.example/sitemap.xml</loc></sitemap></sitemapindex>").encode()
        child = (f"<urlset><url><loc>https://{host}/news/a</loc><lastmod>2026-10-22</lastmod></url>"
                 f"<url><loc>https://{host}/private/no</loc></url></urlset>").encode()
        fetcher = FakeFetcher({f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                               f"https://{host}/sitemap-index.xml": (index, "application/xml"),
                               f"https://{host}/sitemap-news.xml": (gzip.compress(child), "application/octet-stream"),
                               f"https://{host}/news/a": article("Историческая новость")})
        self.assertEqual(self.run_collect(fetcher, max_requests=2)["new"], 0)
        self.assertEqual(self.run_collect(fetcher, max_requests=5)["new"], 1)
        self.assertEqual(fetcher.calls.count(f"https://{host}/sitemap-index.xml"), 1)
        self.assertFalse(any("evil" in url or "/private/" in url for url in fetcher.calls))

    def test_robots_disallow_and_failed_urls_are_not_retried(self):
        host = "news.example.test"
        self.write_sources([source(host)])
        fetcher = FakeFetcher({f"https://{host}/robots.txt": (b"User-agent: *\nDisallow: /news/blocked", "text/plain"),
                               f"https://{host}/rss": rss(host, ["blocked", "failed"]),
                               f"https://{host}/news/failed": CollectionError("http_503")})
        first = self.run_collect(fetcher)
        self.assertEqual(first["errors"], 2)
        self.assertNotIn(f"https://{host}/news/blocked", fetcher.calls)
        second = self.run_collect(fetcher)
        self.assertEqual(second["requests"], 0)
        self.assertEqual(fetcher.calls.count(f"https://{host}/news/failed"), 1)

    def test_source_quotas_keep_two_publishers_in_corpus(self):
        configs = [source("one.example.test", "one"), source("two.example.test", "two")]
        self.write_sources(configs)
        values = {}
        for config in configs:
            host = config["allowed_hosts"][0]
            values[f"https://{host}/robots.txt"] = (b"User-agent: *\nAllow: /", "text/plain")
            values[f"https://{host}/rss"] = rss(host, ["first", "second"])
            for name in ["first", "second"]:
                values[f"https://{host}/news/{name}"] = article("Новость " + host + name, suffix=host + name)
        result = self.run_collect(FakeFetcher(values), max_articles=10, max_source_articles=1)
        self.assertEqual(result["new"], 2)
        self.assertEqual(set(corpus_summary(self.database)["sources"]), {"one", "two"})
        self.assertEqual(result["pending"], 2)

    def test_reader_can_export_while_collector_holds_its_lock(self):
        from alphich_kb.relevance_ml.corpus import _database
        host = "news.example.test"
        self.write_sources([source(host)])
        fetcher = FakeFetcher({f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                               f"https://{host}/rss": rss(host, ["a"]),
                               f"https://{host}/news/a": article("Отраслевая новость")})
        self.run_collect(fetcher)
        with _database(self.database):
            self.assertEqual(len(list(iter_articles(self.database))), 1)

    def test_window_uses_inclusive_publisher_days_and_genre_is_quarantine_not_label(self):
        host = "news.example.test"
        config = source(host)
        config["excluded_title_patterns"] = [r"(?i)\bкуда поехать\b"]
        config["excluded_article_path_prefixes"] = ["/news/promotions/"]
        self.write_sources([config])
        names = ["old", "lower", "upper", "late", "consumer", "local", "promotions/a"]
        samples = {
            "old": ("Старые новости отраслевого рынка", "2026-09-06T23:30:00+03:00"),
            "lower": ("Первый день допустимого периода", "2026-09-07T00:01:00+03:00"),
            "upper": ("Последний день допустимого периода", "2026-10-21T23:59:00+03:00"),
            "late": ("Событие за верхней границей периода", "2026-10-22T00:10:00+03:00"),
            "consumer": ("Куда поехать отдыхать на выходные", "2026-10-20T12:00:00+03:00"),
            "local": ("В Ульяновске открыли небольшой цех", "2026-10-20T12:00:00+03:00"),
            "promotions/a": ("Промоакция производителя оборудования", "2026-10-20T12:00:00+03:00"),
        }
        values = {f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                  f"https://{host}/rss": rss(host, names)}
        values.update({f"https://{host}/news/{name}": article(title, suffix=name, published_at=date)
                       for name, (title, date) in samples.items()})
        result = self.run_collect(FakeFetcher(values), published_after="2026-09-07", published_before="2026-10-21")
        self.assertEqual(result["new"], 3)
        self.assertEqual(result["quarantined"], 4)
        self.assertEqual(result["labels_created"], 0)
        self.assertEqual(result["total_in_window"], 3)
        accepted = list(iter_articles(self.database))
        self.assertTrue(any("небольшой цех" in row["title"] for row in accepted))
        self.assertTrue(all("grade" not in row for row in accepted))
        self.assertTrue(all(row["source_metadata"]["collection_window"]["calendar_timezone"] == "Europe/Moscow"
                            for row in accepted))
        self.assertTrue(all(row["source_metadata"]["publication_date_precision"] == "timestamp" for row in accepted))
        from contextlib import closing
        with closing(sqlite3.connect(self.database)) as db:
            reasons = [row[0] for row in db.execute("SELECT reason FROM quarantine")]
        self.assertEqual(reasons.count("publication_outside_collection_window"), 2)
        self.assertTrue(any(reason.startswith("excluded_genre_title_pattern:") for reason in reasons))
        self.assertTrue(any(reason.startswith("excluded_genre_path:") for reason in reasons))

    def test_source_exclusions_and_publication_bounds_are_validated_before_fetch(self):
        config = source()
        config["excluded_title_patterns"] = ["["]
        self.write_sources([config])
        with self.assertRaises(KnowledgeError):
            load_sources(self.manifest)
        config["excluded_title_patterns"] = []
        config["excluded_article_path_prefixes"] = ["https://other.example/"]
        self.write_sources([config])
        with self.assertRaises(KnowledgeError):
            load_sources(self.manifest)
        config["excluded_article_path_prefixes"] = []
        self.write_sources([config])
        for after, before in [("2026-02-30", None), ("2026-10-22T00:00:00Z", None), ("2026-10-22", "2026-10-21")]:
            fetcher = FakeFetcher({})
            with self.assertRaises(KnowledgeError):
                self.run_collect(fetcher, published_after=after, published_before=before)
            self.assertEqual(fetcher.calls, [])

    def test_sitemap_lastmod_prioritizes_download_without_becoming_publication_proof(self):
        host = "news.example.test"
        config = source(host)
        config["feeds"] = [{"url": f"https://{host}/sitemap-news.xml", "kind": "sitemap"}]
        self.write_sources([config])
        sitemap = (f"<urlset><url><loc>https://{host}/news/old</loc><lastmod>2020-01-01</lastmod></url>"
                   f"<url><loc>https://{host}/news/new</loc><lastmod>2026-10-21</lastmod></url></urlset>").encode()
        fetcher = FakeFetcher({f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                               f"https://{host}/sitemap-news.xml": (sitemap, "application/xml"),
                               f"https://{host}/news/new": article("Новость без подтверждённой даты", date=False),
                               f"https://{host}/news/old": article("Старый индекс настоящая дата", suffix="old")})
        result = self.run_collect(fetcher, max_requests=3)
        self.assertEqual(result["new"], 0)
        self.assertEqual(result["quarantined"], 1)
        self.assertEqual(fetcher.calls[-1], f"https://{host}/news/new")
        self.assertNotIn(f"https://{host}/news/old", fetcher.calls)
        self.assertEqual(self.run_collect(fetcher)["new"], 1)

    def test_changed_window_reports_existing_snapshots_without_silently_deleting(self):
        host = "news.example.test"
        self.write_sources([source(host)])
        fetcher = FakeFetcher({f"https://{host}/robots.txt": (b"User-agent: *\nAllow: /", "text/plain"),
                               f"https://{host}/rss": rss(host, ["a"]),
                               f"https://{host}/news/a": article("Подтверждённая старая публикация")})
        self.assertEqual(self.run_collect(fetcher)["new"], 1)
        result = self.run_collect(fetcher, published_after="2026-10-21", published_before="2026-10-22")
        self.assertEqual(result["existing_outside_window"], 1)
        self.assertEqual(result["total_in_window"], 0)
        self.assertEqual(result["total"], 1)
        self.assertEqual(len(list(iter_articles(self.database))), 1)

    def test_publisher_specific_reverse_preserves_numbered_sitemap_index_order(self):
        config = source()
        config["reverse_sitemap_articles"] = True
        root = "https://news.example.test"
        index = (f"<sitemapindex><sitemap><loc>{root}/sitemap-1.xml</loc></sitemap>"
                 f"<sitemap><loc>{root}/sitemap-7.xml</loc></sitemap></sitemapindex>").encode()
        items = list(_discover(index, root + "/sitemap.xml", "sitemap", "sitemap", config))
        self.assertEqual([item["url"] for item in items], [root + "/sitemap-1.xml", root + "/sitemap-7.xml"])
        entries = (f"<urlset><url><loc>{root}/news/old</loc></url>"
                   f"<url><loc>{root}/news/new</loc></url></urlset>").encode()
        items = list(_discover(entries, root + "/sitemap-1.xml", "sitemap", "sitemap", config))
        self.assertEqual([item["url"] for item in items], [root + "/news/new", root + "/news/old"])

    def test_budgeted_transport_has_no_hidden_second_ip_attempt(self):
        import time
        from unittest.mock import patch
        from alphich_kb.relevance_ml.corpus import _bounded_request
        with (patch("app.rss_fetch.public_addresses", return_value=["8.8.8.8", "1.1.1.1"]) as addresses,
              patch("socket.create_connection", side_effect=OSError("first address failed")) as connect,
              patch("http.client.HTTPConnection") as connection):
            with self.assertRaises(CollectionError) as raised:
                _bounded_request("https://news.example.test/news/a", {}, time.monotonic() + 10, 2_000_000)
        self.assertEqual(raised.exception.code, "download_failed")
        addresses.assert_called_once_with("news.example.test", 443)
        self.assertEqual(connect.call_count, 1)
        self.assertEqual(connect.call_args.args[0], ("8.8.8.8", 443))
        connection.return_value.close.assert_called_once()

    def test_articles_robots_and_sitemaps_share_one_budgeted_transport(self):
        import time
        from types import SimpleNamespace
        from unittest.mock import Mock, patch
        from alphich_kb.relevance_ml.corpus import _real_fetcher, MAX_SITEMAP_BYTES
        from app import rss_fetch
        config = source()
        config["seeds"] = []
        for kind, path in [("article", "/news/a"), ("robots", "/robots.txt"), ("sitemap", "/sitemap.xml")]:
            url = "https://news.example.test" + path
            budget = SimpleNamespace(deadline=time.monotonic() + 10, before=Mock(), after=Mock())
            response = Download(url, 200, {}, b"ok")
            with (patch("alphich_kb.relevance_ml.corpus._bounded_request", return_value=response) as request,
                  patch("app.rss_fetch._request", side_effect=AssertionError("unbudgeted backend transport"))):
                self.assertIs(_real_fetcher(budget, config, kind).get(url, config["allowed_hosts"]), response)
            request.assert_called_once()
            self.assertEqual(request.call_args.args[3], MAX_SITEMAP_BYTES if kind == "sitemap" else rss_fetch.MAX_BYTES)
            budget.before.assert_called_once()
            budget.after.assert_called_once_with(response)


if __name__ == "__main__":
    unittest.main()
