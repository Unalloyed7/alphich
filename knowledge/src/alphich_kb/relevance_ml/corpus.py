"""Resumable, private public-news corpus collection, independent of production DBs.

Only operator-configured publishers, paths and discovery formats are traversed.
Source topics are collection strata, never relevance labels. Publication dates
come from publisher metadata/feed entries, never sitemap lastmod or fetch time.
"""
from __future__ import annotations

import gzip
import hashlib
import heapq
import io
import json
import math
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from datetime import date as calendar_date, datetime, timezone
from pathlib import Path
from urllib import robotparser
from urllib.parse import urljoin, urlsplit, urlunsplit

from ..common import KnowledgeError, canonical, digest


SOURCE_SCHEMA = "alphich-relevance-sources-v1"
CORPUS_SCHEMA = "alphich-news-corpus-v1"
MAX_SITEMAP_BYTES = 8_000_000
_DISCOVERY_FORMATS = {"rss", "mintrans_html", "eec_html", "trendforce_html", "russoft_html", "archive_links"}


class _Stop(Exception):
    pass


def _deps():
    try:
        from app import rss_fetch, rss_parse
    except ImportError as error:
        raise KnowledgeError("Corpus collection needs backend dependencies and PYTHONPATH=backend:knowledge/src") from error
    return rss_fetch, rss_parse


def _normalize_url(value):
    parts = urlsplit(value)
    # Tracking-only query parameters must not turn the same publication into
    # many requests. Meaningful pagination/id parameters are preserved.
    from urllib.parse import parse_qsl, urlencode
    query = [(key, val) for key, val in parse_qsl(parts.query, keep_blank_values=True)
             if not key.casefold().startswith("utm_") and key.casefold() not in {"fbclid", "gclid", "yclid"}]
    return urlunsplit((parts.scheme, parts.netloc, parts.path or "/", urlencode(query), ""))


def _seed(item):
    if isinstance(item, str):
        return {"url": item, "kind": "sitemap" if urlsplit(item).path.endswith((".xml", ".gz")) else "html",
                "format": "archive_links"}
    if not isinstance(item, dict) or not isinstance(item.get("url"), str):
        raise KnowledgeError("Each corpus seed needs a URL")
    return {"url": item["url"], "kind": item.get("kind", "rss"), "format": item.get("format", "rss")}


def load_sources(path):
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != SOURCE_SCHEMA or not isinstance(raw.get("sources"), list):
        raise KnowledgeError("Unsupported relevance source manifest")
    sources = []
    ids = set()
    for source in raw["sources"]:
        if not isinstance(source, dict) or not isinstance(source.get("id"), str) or source["id"] in ids:
            raise KnowledgeError("Corpus source IDs must be unique strings")
        ids.add(source["id"])
        if source.get("status", "available") != "available" or source.get("enabled", True) is not True:
            continue
        hosts = source.get("allowed_hosts")
        prefixes = source.get("article_path_prefixes")
        if (not isinstance(hosts, list) or not hosts or any(not isinstance(host, str) or not host or "/" in host for host in hosts)
                or not isinstance(prefixes, list) or not prefixes or any(not isinstance(value, str) or not value.startswith("/") for value in prefixes)):
            raise KnowledgeError("Each source requires exact allowed_hosts and article_path_prefixes")
        item = dict(source)
        item["allowed_hosts"] = frozenset(host.encode("idna").decode("ascii").lower().rstrip(".") for host in hosts)
        item["seeds"] = [_seed(seed) for seed in [*source.get("feeds", []), *source.get("archive_urls", [])]]
        if not item["seeds"]:
            continue
        item["min_interval_seconds"] = max(0.5, float(source.get("min_interval_seconds", 1.0)))
        if not math.isfinite(item["min_interval_seconds"]) or item["min_interval_seconds"] > 60:
            raise KnowledgeError("Publisher interval must be finite and at most 60 seconds")
        for field in ("article_url_regex", "pagination_url_regex", "sitemap_url_regex"):
            if item.get(field) is not None:
                if not isinstance(item[field], str) or len(item[field]) > 1000:
                    raise KnowledgeError("Invalid publisher URL pattern")
                try:
                    re.compile(item[field])
                except re.error as error:
                    raise KnowledgeError("Invalid publisher URL pattern") from error
        patterns = source.get("excluded_title_patterns", [])
        excluded_paths = source.get("excluded_article_path_prefixes", [])
        if (not isinstance(patterns, list) or len(patterns) > 50
                or any(not isinstance(value, str) or not value or len(value) > 1000 for value in patterns)):
            raise KnowledgeError("Invalid excluded title patterns")
        try:
            item["_excluded_title_patterns"] = [(value, re.compile(value)) for value in patterns]
        except re.error as error:
            raise KnowledgeError("Invalid excluded title pattern") from error
        if (not isinstance(excluded_paths, list) or len(excluded_paths) > 50
                or any(not isinstance(value, str) or not value.startswith("/") or len(value) > 1000
                       or "?" in value or "#" in value for value in excluded_paths)):
            raise KnowledgeError("Invalid excluded article path prefixes")
        if type(source.get("reverse_sitemap_articles", False)) is not bool:
            raise KnowledgeError("reverse_sitemap_articles must be boolean")
        for seed in item["seeds"]:
            if seed["kind"] not in {"rss", "html", "sitemap"} or (seed["kind"] != "sitemap" and seed["format"] not in _DISCOVERY_FORMATS):
                raise KnowledgeError("Unsupported corpus discovery format")
            _allowed(seed["url"], item, seed["kind"], seed=True)
        sources.append(item)
    if not sources:
        raise KnowledgeError("Manifest contains no enabled, available source seeds")
    return sources


def _allowed(url, source, kind, *, seed=False):
    fetch, _ = _deps()
    url = _normalize_url(fetch.public_url(url, source["allowed_hosts"]))
    path = urlsplit(url).path
    if kind == "robots":
        allowed = path == "/robots.txt"
    elif kind == "article":
        allowed = any(path.startswith(prefix) for prefix in source["article_path_prefixes"])
        if source.get("article_url_regex"):
            allowed = allowed and bool(re.fullmatch(source["article_url_regex"], url))
    elif seed:
        allowed = True  # The URL is explicitly operator-configured in the manifest.
    elif kind == "sitemap":
        prefixes = source.get("sitemap_path_prefixes", [])
        allowed = ((bool(prefixes) and any(path.startswith(prefix) for prefix in prefixes))
                   or (not prefixes and "sitemap" in path.casefold() and path.endswith((".xml", ".gz"))))
        if source.get("sitemap_url_regex"):
            allowed = bool(re.fullmatch(source["sitemap_url_regex"], url))
    else:
        # A publisher may canonicalize www aliases. Exact hosts were validated
        # above; only an already configured discovery path/query can redirect.
        configured = {(urlsplit(value["url"]).path, urlsplit(value["url"]).query) for value in source["seeds"]}
        allowed = (path, urlsplit(url).query) in configured or bool(source.get("pagination_url_regex") and re.fullmatch(source["pagination_url_regex"], url))
    if not allowed:
        raise fetch.CollectionError("source_path_not_allowed")
    return url


@contextmanager
def _database(path):
    import fcntl
    destination = Path(path)
    if destination.is_symlink() or Path(str(destination) + ".lock").is_symlink():
        raise KnowledgeError("Corpus database and lock must not be symlinks")
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock = os.open(str(destination) + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise KnowledgeError("Another collector is using this corpus") from error
        created = not destination.exists()
        db = sqlite3.connect(destination, timeout=10)
        os.chmod(destination, 0o600)
        db.row_factory = sqlite3.Row
        db.executescript("""
          CREATE TABLE IF NOT EXISTS metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS queue(
            url TEXT PRIMARY KEY, source_id TEXT NOT NULL, kind TEXT NOT NULL, format TEXT NOT NULL,
            title TEXT, published_at TEXT, depth INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
            error TEXT, processed_at TEXT);
          CREATE INDEX IF NOT EXISTS queue_pending ON queue(state,kind);
          CREATE TABLE IF NOT EXISTS hosts(
            host TEXT PRIMARY KEY,last_request REAL NOT NULL DEFAULT 0,
            robots_at REAL,robots_body TEXT,robots_status TEXT);
          CREATE TABLE IF NOT EXISTS publisher_holds(
            source_id TEXT PRIMARY KEY,retry_at REAL NOT NULL,reason TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS articles(
            article_id TEXT PRIMARY KEY,url TEXT NOT NULL UNIQUE,source_id TEXT NOT NULL,
            title TEXT NOT NULL,title_norm TEXT NOT NULL,text TEXT NOT NULL,
            content_hash TEXT NOT NULL UNIQUE,cluster_id TEXT NOT NULL,signature TEXT NOT NULL,
            published_at TEXT NOT NULL,fetched_at TEXT NOT NULL,source_metadata TEXT NOT NULL);
          CREATE INDEX IF NOT EXISTS article_title ON articles(title_norm);
          CREATE TABLE IF NOT EXISTS quarantine(
            url TEXT PRIMARY KEY,source_id TEXT NOT NULL,title TEXT,text TEXT,
            fetched_at TEXT NOT NULL,reason TEXT NOT NULL);
          CREATE TABLE IF NOT EXISTS anchors(anchor TEXT NOT NULL,article_id TEXT NOT NULL,PRIMARY KEY(anchor,article_id));
          CREATE INDEX IF NOT EXISTS anchors_article ON anchors(article_id);
        """)
        # Additive migration preserves old resumable queues and article snapshots.
        if "priority_at" not in {row[1] for row in db.execute("PRAGMA table_info(queue)")}:
            db.execute("ALTER TABLE queue ADD COLUMN priority_at REAL NOT NULL DEFAULT 0")
            for row in db.execute("SELECT url,published_at FROM queue WHERE published_at IS NOT NULL").fetchall():
                db.execute("UPDATE queue SET priority_at=? WHERE url=?", (_priority_epoch(row[1]), row[0]))
        for column in ("published_at", "publication_evidence"):
            if column not in {row[1] for row in db.execute("PRAGMA table_info(quarantine)")}:
                db.execute("ALTER TABLE quarantine ADD COLUMN " + column + " TEXT")
        version = db.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
        if version is not None and version[0] != CORPUS_SCHEMA:
            db.close()
            raise KnowledgeError("Unsupported corpus database schema")
        db.execute("INSERT OR IGNORE INTO metadata VALUES('schema_version',?)", (CORPUS_SCHEMA,))
        db.commit()
        try:
            yield db
        finally:
            db.commit()
            db.close()
    finally:
        os.close(lock)


def _utc(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat()


def _priority_epoch(value):
    """Discovery hint only; a sitemap lastmod never becomes published_at."""
    _, parse = _deps()
    normalized = parse.source_date(value) if isinstance(value, str) else None
    try:
        if not normalized:
            # Sitemap lastmod commonly contains a day only. This UTC value is
            # solely a sorting hint, never passed to publication-date parsing.
            if isinstance(value, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
                return datetime.fromisoformat(value).replace(tzinfo=timezone.utc).timestamp()
            return 0.0
        parsed = datetime.fromisoformat(normalized)
        return parsed.timestamp() if parsed.tzinfo is not None else 0.0
    except (ValueError, OverflowError):
        return 0.0


def _collection_window(published_after, published_before):
    bounds = []
    for value in (published_after, published_before):
        if value is None:
            bounds.append(None)
            continue
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise KnowledgeError("Publication bounds must be ISO dates YYYY-MM-DD")
        try:
            bounds.append(calendar_date.fromisoformat(value))
        except ValueError as error:
            raise KnowledgeError("Invalid publication bound date") from error
    if all(bounds) and bounds[0] > bounds[1]:
        raise KnowledgeError("Publication lower bound must not exceed upper bound")
    return tuple(bounds)


def _within_window(publication, bounds):
    from zoneinfo import ZoneInfo
    parsed = datetime.fromisoformat(publication)
    if parsed.tzinfo is None:
        return False
    day = parsed.astimezone(ZoneInfo("Europe/Moscow")).date()
    return (bounds[0] is None or day >= bounds[0]) and (bounds[1] is None or day <= bounds[1])


def _genre_exclusion(source, title, url):
    for prefix in source.get("excluded_article_path_prefixes", []):
        if urlsplit(url).path.startswith(prefix):
            return "excluded_genre_path:" + prefix
    for raw, pattern in source.get("_excluded_title_patterns", []):
        if pattern.search(title):
            return "excluded_genre_title_pattern:" + raw
    return None


def _date_precision(evidence):
    if any(item.get("date_only_available_at_day_end") for item in evidence):
        return "day"
    if any(isinstance(item.get("raw"), str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", item["raw"]) for item in evidence):
        return "day"
    return "timestamp" if any(item.get("raw") for item in evidence) else "unknown"


class _Budget:
    def __init__(self, db, source_map, max_requests, max_bytes, max_runtime, clock):
        self.db, self.sources = db, source_map
        self.max_requests, self.max_bytes = max_requests, max_bytes
        self.deadline = time.monotonic() + max_runtime
        self.clock, self.requests, self.bytes, self.received = clock, 0, 0, 0
        self.pending_limit = 0

    def before(self, url, source, response_limit=None):
        fetch, _ = _deps()
        hold = self.db.execute("SELECT retry_at FROM publisher_holds WHERE source_id=?", (source["id"],)).fetchone()
        if hold and hold[0] > self.clock():
            raise _Stop("publisher_cooldown")
        # Reserve the fetcher's full bounded response before making an HTTP
        # request, so even a final unexpectedly large response cannot exceed
        # the invocation's byte cap.
        response_limit = response_limit or fetch.MAX_BYTES
        if self.requests >= self.max_requests or self.bytes + response_limit > self.max_bytes:
            raise _Stop("request_or_byte_budget")
        if time.monotonic() >= self.deadline:
            raise _Stop("runtime_budget")
        host = urlsplit(url).hostname
        row = self.db.execute("SELECT last_request FROM hosts WHERE host=?", (host,)).fetchone()
        interval = max(source["min_interval_seconds"], source.get("_robots_intervals", {}).get(host, 0))
        remaining = interval - (self.clock() - (row[0] if row else 0))
        if remaining > 0:
            if time.monotonic() + remaining >= self.deadline:
                raise _Stop("runtime_budget")
            time.sleep(remaining)
        self.requests += 1
        self.bytes += response_limit
        self.pending_limit = response_limit
        self.db.execute("INSERT INTO hosts(host,last_request) VALUES(?,?) ON CONFLICT(host) DO UPDATE SET last_request=excluded.last_request", (host, self.clock()))
        self.db.commit()

    def after(self, response):
        self.bytes += len(response.body) - self.pending_limit
        self.received += len(response.body)
        self.pending_limit = 0


def _hold_publisher(budget, source, reason, delay):
    budget.db.execute("INSERT INTO publisher_holds VALUES(?,?,?) ON CONFLICT(source_id) DO UPDATE SET "
                      "retry_at=MAX(retry_at,excluded.retry_at),reason=excluded.reason",
                      (source["id"], budget.clock() + delay, reason))
    budget.db.commit()


def _observe_backoff(response, budget, source):
    if response.status not in {429, 503}:
        return
    from email.utils import parsedate_to_datetime
    raw = response.headers.get("retry-after", "").strip()
    delay = 3600 if response.status == 429 else 900
    try:
        if raw.isdigit() and len(raw) <= 12:
            delay = max(60, int(raw))
        elif raw:
            parsed = parsedate_to_datetime(raw)
            if parsed.tzinfo is not None:
                delay = max(60, parsed.timestamp() - budget.clock())
    except (ValueError, TypeError, OverflowError):
        pass
    _hold_publisher(budget, source, "http_" + str(response.status), delay)
    # Preserve the current URL as pending. A later bounded invocation may
    # retry it only after the persisted publisher deadline, including aliases.
    raise _Stop("publisher_cooldown")


def _real_fetcher(budget, source, kind, *, seed=False):
    """PublicFetcher transport, adding path validation before every redirect hop."""
    fetch, _ = _deps()

    class PathBoundPublicFetcher(fetch.PublicFetcher):
        def get(self, url, hosts, validators=None):
            url = _allowed(url, source, kind, seed=seed)
            headers = {"User-Agent": fetch.USER_AGENT, "Accept-Encoding": "identity", "Accept": "*/*"}
            for _ in range(4):
                limit = MAX_SITEMAP_BYTES if kind == "sitemap" else fetch.MAX_BYTES
                budget.before(url, source, limit)
                result = _bounded_request(url, headers, min(time.monotonic() + fetch.TIMEOUT, budget.deadline), limit)
                budget.after(result)
                _observe_backoff(result, budget, source)
                if result.status not in {301, 302, 303, 307, 308}:
                    return result
                target = _allowed(urljoin(url, result.headers.get("location", "")), source, kind)
                if urlsplit(url).scheme == "https" and urlsplit(target).scheme != "https":
                    raise fetch.CollectionError("https_downgrade_blocked")
                if kind != "robots" and not _robots_allows(target, source, budget.db, budget, None):
                    raise fetch.CollectionError("redirect_robots_disallowed_or_unavailable")
                url = target
            raise fetch.CollectionError("redirect_limit")

    return PathBoundPublicFetcher()


def _bounded_request(url, headers, deadline, limit):
    """One validated-address attempt per budgeted HTTP hop, with pinned-IP TLS.

    All DNS answers are validated before selecting one address. Failure does
    not trigger hidden IP retries beyond the caller's request/rate budget.
    Articles/robots retain the 2MB cap; only sitemaps receive the 8MB cap.
    Production rss_fetch and its transport remain unchanged.
    """
    import http.client
    import socket
    import ssl
    fetch, _ = _deps()
    parts = urlsplit(url)
    port = 443 if parts.scheme == "https" else 80
    addresses = fetch.public_addresses(parts.hostname, port)
    for address in addresses[:1]:
        wire = None
        connection = http.client.HTTPConnection(parts.hostname, port, timeout=min(10, max(0.1, deadline - time.monotonic())))
        try:
            wire = socket.create_connection((address, port), timeout=connection.timeout)
            if parts.scheme == "https":
                context = ssl.create_default_context()
                context.set_alpn_protocols(["http/1.1"])
                wire = context.wrap_socket(wire, server_hostname=parts.hostname)
            connection.sock = wire
            connection.request("GET", urlunsplit(("", "", parts.path or "/", parts.query, "")), headers=headers)
            response = connection.getresponse()
            meta = {key.lower(): value for key, value in response.getheaders()}
            if response.status in {301, 302, 303, 307, 308}:
                return fetch.Download(url, response.status, meta, b"")
            if response.status != 200:
                return fetch.Download(url, response.status, meta, b"")
            length = meta.get("content-length")
            if length is not None and (not length.isdigit() or int(length) > limit):
                raise fetch.CollectionError("response_too_large")
            body = bytearray()
            while len(body) <= limit:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise fetch.CollectionError("download_timeout")
                wire.settimeout(min(10, remaining))
                block = response.read1(min(65536, limit + 1 - len(body)))
                if not block:
                    break
                body.extend(block)
            if len(body) > limit or (length is not None and len(body) != int(length)):
                raise fetch.CollectionError("response_too_large_or_incomplete")
            encoding = meta.get("content-encoding", "identity").casefold()
            if encoding == "gzip":
                with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                    body = stream.read(limit + 1)
            elif encoding != "identity":
                raise fetch.CollectionError("unsupported_compression")
            if len(body) > limit:
                raise fetch.CollectionError("response_too_large")
            return fetch.Download(url, response.status, meta, bytes(body))
        except fetch.CollectionError:
            raise
        except (OSError, EOFError, http.client.HTTPException, ValueError):
            raise fetch.CollectionError("download_failed") from None
        finally:
            connection.close()
            if wire is not None:
                wire.close()
    raise fetch.CollectionError("download_failed")


def _get(url, source, kind, budget, fetcher, *, seed=False):
    url = _allowed(url, source, kind, seed=seed)
    if fetcher is None:
        return _real_fetcher(budget, source, kind, seed=seed).get(url, source["allowed_hosts"])
    budget.before(url, source, MAX_SITEMAP_BYTES if kind == "sitemap" else None)
    result = fetcher.get(url, source["allowed_hosts"])
    budget.after(result)
    _observe_backoff(result, budget, source)
    _allowed(result.url, source, kind, seed=seed)
    return result


def _robots_allows(url, source, db, budget, fetcher):
    fetch, _ = _deps()
    host = urlsplit(url).hostname
    row = db.execute("SELECT robots_at,robots_body,robots_status FROM hosts WHERE host=?", (host,)).fetchone()
    if row is None or row[0] is None or budget.clock() - row[0] > 86_400:
        robots_url = urlunsplit((urlsplit(url).scheme, host, "/robots.txt", "", ""))
        status, body = "blocked", ""
        try:
            result = _get(robots_url, source, "robots", budget, fetcher)
            if result.status == 200 and len(result.body) <= 512_000:
                body = result.body.decode("utf-8", errors="replace")
                # Error pages served as HTTP200 are not a valid robots policy.
                status = "ok" if not body.strip() or re.search(r"(?im)^\s*user-agent\s*:", body) else "invalid"
            elif result.status == 404:
                status = "absent"
        except fetch.CollectionError as error:
            if error.code == "http_404":
                status = "absent"
            else:
                status = "error:" + error.code
        db.execute("UPDATE hosts SET robots_at=?,robots_body=?,robots_status=? WHERE host=?", (budget.clock(), body, status, host))
        db.commit()
    else:
        body, status = row[1] or "", row[2]
    if status == "absent":
        return True
    if status != "ok":
        return False
    parser = robotparser.RobotFileParser()
    parser.parse(body.splitlines())
    interval = parser.crawl_delay(fetch.USER_AGENT) or 0
    rate = parser.request_rate(fetch.USER_AGENT)
    if rate and rate.requests > 0:
        interval = max(interval, rate.seconds / rate.requests)
    source.setdefault("_robots_intervals", {})[host] = interval
    return parser.can_fetch(fetch.USER_AGENT, url)


def _xml(body):
    fetch, _ = _deps()
    if body.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(body)) as stream:
                body = stream.read(MAX_SITEMAP_BYTES + 1)
        except (OSError, EOFError) as error:
            raise fetch.CollectionError("invalid_sitemap_compression") from error
    if len(body) > MAX_SITEMAP_BYTES:
        raise fetch.CollectionError("sitemap_too_large")
    from defusedxml import ElementTree
    from defusedxml.common import DefusedXmlException
    from xml.etree.ElementTree import ParseError
    try:
        return ElementTree.fromstring(body, forbid_dtd=True, forbid_entities=True, forbid_external=True)
    except (ParseError, DefusedXmlException, ValueError) as error:
        raise fetch.CollectionError("invalid_sitemap_xml") from error


def _html(body):
    from lxml import etree, html
    try:
        declared = re.search(br"charset\s*=\s*['\"]?([a-zA-Z0-9_-]+)", body[:8000])
        encoding = declared[1].decode("ascii").casefold() if declared else "utf-8"
        encoding = encoding if encoding in {"utf-8", "utf8", "windows-1251", "cp1251", "koi8-r"} else "utf-8"
        return html.fromstring(body.decode(encoding, errors="replace"), parser=html.HTMLParser(no_network=True, recover=True))
    except (etree.ParserError, ValueError) as error:
        fetch, _ = _deps()
        raise fetch.CollectionError("invalid_article_html") from error


def _publisher_local_date(value):
    """Inspected Russian publishers' explicit local dates, never relative dates.

    Date-only values are conservatively available at the end of the publisher
    day. Europe/Moscow records historical DST; no host-machine timezone is used.
    """
    from zoneinfo import ZoneInfo
    if not isinstance(value, str):
        return None
    value = " ".join(value.casefold().split()).replace(" в ", " ")
    months = {name: str(index) for index, name in enumerate("января февраля марта апреля мая июня июля августа сентября октября ноября декабря".split(), 1)}
    for name, number in months.items():
        value = value.replace(name, number)
    match = re.fullmatch(r"(\d{1,2})[ .](\d{1,2})[ .](\d{4})(?:[ ,]+(\d{1,2}):(\d{2}))?\.?", value)
    iso = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", value)
    try:
        if match:
            day, month, year = map(int, match.group(1, 2, 3))
            hour, minute, second = (int(match[4]), int(match[5]), 0) if match[4] else (23, 59, 59)
        elif iso:
            year, month, day = map(int, iso.groups())
            hour, minute, second = 23, 59, 59
        else:
            return None
        return datetime(year, month, day, hour, minute, second, tzinfo=ZoneInfo("Europe/Moscow")).astimezone(timezone.utc).isoformat()
    except ValueError:
        return None


def _date_and_title(body, fallback_title, fallback_date, source_id=None):
    _, parse = _deps()
    document = _html(body)
    dates = set()
    evidence = []
    for raw in document.xpath('//head/meta[@property="article:published_time"]/@content'):
        if (date := parse.source_date(raw)):
            dates.add(date)
            evidence.append({"selector": "meta.article:published_time", "raw": raw, "normalized": date})
    for value in document.xpath('//time[@itemprop="datePublished"]/@datetime | //meta[@itemprop="datePublished"]/@content'):
        if (date := parse.source_date(value)):
            dates.add(date)
            evidence.append({"selector": "itemprop.datePublished", "raw": value, "normalized": date})
    local_values = []
    for script in document.xpath('//script[@type="application/ld+json"]/text()')[:30]:
        if len(script) > 200_000:
            continue
        try:
            value = json.loads(script)
        except (ValueError, RecursionError):
            continue
        nodes = value if isinstance(value, list) else [value]
        for node in nodes:
            if isinstance(node, dict) and isinstance(node.get("@graph"), list):
                nodes = nodes + node["@graph"][:100]
        for node in nodes:
            if not isinstance(node, dict):
                continue
            kinds = node.get("@type", [])
            kinds = [kinds] if isinstance(kinds, str) else kinds
            if isinstance(kinds, list) and set(kinds) & {"Article", "NewsArticle", "BlogPosting", "Report"}:
                raw_date = node.get("datePublished")
                if (date := parse.source_date(raw_date)):
                    dates.add(date)
                    evidence.append({"selector": "jsonld.datePublished", "raw": raw_date, "normalized": date})
                elif source_id == "stroygaz" and isinstance(raw_date, str):
                    local_values.append(("jsonld.datePublished", raw_date))
    if not dates:
        if source_id == "agroinvestor":
            local_values += [("time[itemprop=datePublished]@datatime", value) for value in document.xpath('//time[@itemprop="datePublished"]/@datatime')]
        elif source_id == "autostat":
            local_values += [("span[itemprop=datePublished]@content", value) for value in document.xpath('//span[@itemprop="datePublished"]/@content')]
        elif source_id == "fns":
            for node in document.xpath('//*[contains(concat(" ",normalize-space(@class)," ")," single-news__time ")]'):
                value = " ".join(node.itertext()).strip()
                if value.startswith("Дата публикации:"):
                    local_values.append(("single-news__time.publication", value.split(":", 1)[1].strip()))
        elif source_id == "mintrans":
            for node in document.xpath('//p[contains(concat(" ",normalize-space(@class)," ")," article-inf-date ")]'):
                value = " ".join(node.itertext()).strip()
                if value.casefold().startswith("опубликовано "):
                    local_values.append(("p.article-inf-date.publication", value[len("Опубликовано "):].strip()))
        elif source_id == "minzdrav":
            for node in document.xpath('//p[contains(concat(" ",normalize-space(@class)," ")," timestamps ")]'):
                value = " ".join(" ".join(node.itertext()).split())
                match = re.match(r"Материал опубликован\s+(.+?)\.(?:\s|$)", value)
                if match:
                    local_values.append(("p.timestamps.published", match[1]))
        for selector, value in local_values:
            if (date := _publisher_local_date(value)):
                dates.add(date)
                evidence.append({"selector": selector, "raw": value, "normalized": date,
                                 "timezone": "Europe/Moscow", "date_only_available_at_day_end": not bool(re.search(r"\d{1,2}:\d{2}", value))})
    # Conflicting article dates are quarantined, not replaced with a feed date.
    publication = next(iter(dates)) if len(dates) == 1 else (fallback_date if not dates else None)
    titles = document.xpath('//head/meta[@property="og:title"]/@content')
    titles += [" ".join(node.itertext()) for node in document.xpath("//h1")[:1]]
    title = next((" ".join(value.split()) for value in titles if isinstance(value, str) and value.strip()), fallback_title or "")
    if not dates and fallback_date:
        evidence.append({"selector": "publisher_feed_or_listing", "normalized": fallback_date})
    return publication, title, evidence


def _discover(body, url, kind, format, source):
    """Yield queue entries; no publication dates are inferred from sitemap lastmod."""
    fetch, parse = _deps()
    if kind == "sitemap":
        root = _xml(body)
        name = parse.tag(root)
        if name not in {"sitemapindex", "urlset"}:
            raise fetch.CollectionError("unsupported_sitemap_format")
        nodes = list(root)[:50_000]
        # Publisher-specific order is explicit; numbered indexes are never
        # globally reversed (Autostat and RUSSOFT use opposite numbering).
        if name == "urlset" and source.get("reverse_sitemap_articles", False):
            nodes.reverse()
        for node in nodes:
            location = parse.child_text(node, "loc")
            if location:
                yield {"url": urljoin(url, location), "kind": "sitemap" if name == "sitemapindex" else "article",
                       "format": "sitemap", "title": None, "published_at": None,
                       "priority_at": parse.child_text(node, "lastmod")}
        return
    if format != "archive_links":
        for item in parse.discover_items(body, url, format=format):
            yield {"url": item.url, "kind": "article", "format": "html", "title": item.title, "published_at": item.published_at}
    document = _html(body) if kind == "html" else None
    if document is not None:
        for node in document.xpath("//a[@href]")[:20_000]:
            target = urljoin(url, node.get("href"))
            if source.get("pagination_url_regex") and re.fullmatch(source["pagination_url_regex"], target):
                yield {"url": target, "kind": "html", "format": format, "title": None, "published_at": None}
            elif format == "archive_links" and source.get("article_url_regex") and re.fullmatch(source["article_url_regex"], target):
                yield {"url": target, "kind": "article", "format": "html", "title": " ".join(node.itertext()).strip(), "published_at": None}


def _signature(text):
    tokens = re.findall(r"\w+", text.casefold().replace("ё", "е"))
    shingles = {hashlib.blake2b(" ".join(tokens[index:index + 5]).encode(), digest_size=8).hexdigest()
                for index in range(max(1, len(tokens) - 4))}
    return heapq.nsmallest(128, shingles)


def _cluster(db, title, signature):
    title_norm = " ".join(re.findall(r"\w+", title.casefold().replace("ё", "е")))
    row = db.execute("SELECT cluster_id FROM articles WHERE title_norm=? LIMIT 1", (title_norm,)).fetchone()
    if row:
        return title_norm, row[0]
    anchors = signature[:8]
    if anchors:
        placeholders = ",".join("?" for _ in anchors)
        candidates = db.execute(f"SELECT DISTINCT a.cluster_id,a.signature FROM anchors k JOIN articles a USING(article_id) WHERE k.anchor IN ({placeholders}) LIMIT 100", anchors).fetchall()
        current = set(signature)
        for candidate in candidates:
            previous = set(json.loads(candidate["signature"]))
            if len(current & previous) / max(1, len(current | previous)) >= 0.82:
                return title_norm, candidate["cluster_id"]
    return title_norm, "cluster_" + digest({"title": title_norm, "signature": signature})[:24]


def _enqueue(db, source, item, *, depth, max_queue_urls, seed=False):
    fetch, _ = _deps()
    try:
        url = _allowed(item["url"], source, item["kind"], seed=seed)
    except fetch.CollectionError:
        return False
    if depth > 12:
        return False
    priority = _priority_epoch(item.get("published_at") or item.get("priority_at"))
    if db.execute("SELECT 1 FROM queue WHERE url=?", (url,)).fetchone():
        # Rediscovery may supply a newer ranking hint, never publication proof.
        db.execute("UPDATE queue SET priority_at=MAX(priority_at,?) WHERE url=? AND state='pending'", (priority, url))
        return False
    if db.execute("SELECT COUNT(*) FROM queue").fetchone()[0] >= max_queue_urls:
        raise _Stop("queue_size_budget")
    db.execute("INSERT INTO queue(url,source_id,kind,format,title,published_at,depth,priority_at) VALUES(?,?,?,?,?,?,?,?)",
               (url, source["id"], item["kind"], item["format"], item.get("title"), item.get("published_at"), depth, priority))
    return True


def corpus_summary(db_path):
    with _database(db_path) as db:
        counts = {row[0]: row[1] for row in db.execute("SELECT state,COUNT(*) FROM queue GROUP BY state")}
        return {"articles": db.execute("SELECT COUNT(*) FROM articles").fetchone()[0],
                "queue": counts, "sources": {row[0]: row[1] for row in db.execute("SELECT source_id,COUNT(*) FROM articles GROUP BY source_id")}}


def iter_articles(db_path, *, limit=None):
    """Read immutable accepted article snapshots, including publisher provenance."""
    destination = Path(db_path)
    if not destination.is_file() or destination.is_symlink():
        raise KnowledgeError("Corpus reader needs an existing regular database")
    if limit is not None and (type(limit) is not int or limit < 1):
        raise KnowledgeError("Corpus limit must be a positive integer")
    # Read-only keyset pages release their SQLite read lock before yielding.
    # Embedding/export may run while a worker continues committing articles.
    db = sqlite3.connect(destination.resolve().as_uri() + "?mode=ro", uri=True, timeout=10)
    db.row_factory = sqlite3.Row
    try:
        cutoff = db.execute("SELECT MAX(fetched_at) FROM articles").fetchone()[0]
        last, emitted = "", 0
        while cutoff and (limit is None or emitted < limit):
            size = min(128, limit - emitted) if limit is not None else 128
            rows = db.execute("SELECT * FROM articles WHERE article_id>? AND fetched_at<=? ORDER BY article_id LIMIT ?",
                              (last, cutoff, size)).fetchall()
            if not rows:
                break
            for row in rows:
                value = dict(row)
                value["source_metadata"] = json.loads(value["source_metadata"])
                value.pop("signature")
                value.pop("title_norm")
                value["source_version"] = value["content_hash"]
                emitted += 1
                last = row["article_id"]
                yield value
    finally:
        db.close()


def collect(manifest_path, db_path, *, max_articles=100, max_requests=300, max_total_bytes=100_000_000,
            max_runtime_seconds=600, goal=10_000, max_queue_urls=200_000, max_source_articles=2500,
            published_after=None, published_before=None, fetcher=None, clock=None):
    """Collect one bounded resumable batch; rate-limited URLs wait for cooldown.

    max_articles bounds newly accepted unique articles, goal bounds the complete
    corpus. Seed pages are fetched once, and pending discovered URLs survive
    restarts. An operator can add new archive seeds in a subsequent manifest.
    fetcher is a test seam; normal runs use SSRF-safe PublicFetcher transport.
    Publication bounds are inclusive YYYY-MM-DD calendar days in Europe/Moscow.
    Existing snapshots are preserved and counted separately when outside a new
    window; only newly accepted articles are constrained by that invocation.
    """
    fetch, parse = _deps()
    for value in (max_articles, max_requests, max_total_bytes, max_runtime_seconds, goal, max_queue_urls, max_source_articles):
        if type(value) is not int or value < 1:
            raise KnowledgeError("Corpus budgets must be positive integers")
    if (max_articles > 10_000 or max_requests > 100_000 or max_total_bytes < fetch.MAX_BYTES
            or max_total_bytes > 2_000_000_000 or max_runtime_seconds > 3600 or goal > 100_000 or max_queue_urls > 1_000_000):
        raise KnowledgeError("Corpus budget exceeds supported bounds")
    sources = {source["id"]: source for source in load_sources(manifest_path)}
    bounds = _collection_window(published_after, published_before)
    clock = clock or time.time
    added, errors, quarantined, duplicate, stopped = 0, 0, 0, 0, "queue_exhausted"
    consecutive_errors = {}
    with _database(db_path) as db:
        budget = _Budget(db, sources, max_requests, max_total_bytes, max_runtime_seconds, clock)
        for source in sources.values():
            for seed in source["seeds"]:
                _enqueue(db, source, seed, depth=0, max_queue_urls=max_queue_urls, seed=True)
        db.commit()
        source_attempts = {row[0]: row[1] for row in db.execute("SELECT source_id,COUNT(*) FROM queue WHERE state<>'pending' GROUP BY source_id")}
        while added < max_articles and db.execute("SELECT COUNT(*) FROM articles").fetchone()[0] < goal:
            if time.monotonic() >= budget.deadline:
                stopped = "runtime_budget"
                break
            counts = {row[0]: row[1] for row in db.execute("SELECT source_id,COUNT(*) FROM articles GROUP BY source_id")}
            pending_sources = {row[0] for row in db.execute("SELECT DISTINCT source_id FROM queue WHERE state='pending'")}
            eligible = [sid for sid in pending_sources if sid in sources and counts.get(sid, 0) < max_source_articles]
            held = {row[0] for row in db.execute("SELECT source_id FROM publisher_holds WHERE retry_at>?", (clock(),))}
            waiting_for_publisher = bool(set(eligible) & held)
            eligible = [sid for sid in eligible if sid not in held]
            if not eligible:
                stopped = "publisher_cooldown" if waiting_for_publisher else "source_quotas_or_queue_exhausted"
                break
            sid = min(eligible, key=lambda value: (source_attempts.get(value, 0), value))
            row = db.execute("SELECT * FROM queue WHERE state='pending' AND source_id=? ORDER BY CASE kind WHEN 'article' THEN 0 ELSE 1 END,priority_at DESC,depth,rowid LIMIT 1", (sid,)).fetchone()
            if row is None:
                break
            source = sources.get(row["source_id"])
            state, error = "done", None
            try:
                if source is None:
                    raise fetch.CollectionError("source_disabled_or_removed")
                if not _robots_allows(row["url"], source, db, budget, fetcher):
                    raise fetch.CollectionError("robots_disallowed_or_unavailable")
                result = _get(row["url"], source, row["kind"], budget, fetcher, seed=row["depth"] == 0)
                if result.status != 200:
                    raise fetch.CollectionError("http_" + str(result.status))
                if len(result.body) > (MAX_SITEMAP_BYTES if row["kind"] == "sitemap" else fetch.MAX_BYTES):
                    raise fetch.CollectionError("response_too_large")
                if row["kind"] != "article":
                    for discovered in _discover(result.body, result.url, row["kind"], row["format"], source):
                        _enqueue(db, source, discovered, depth=row["depth"] + 1, max_queue_urls=max_queue_urls)
                else:
                    text = parse.article_text(result.body, result.url, result.headers.get("content-type", ""))
                    date, title, date_evidence = _date_and_title(result.body, row["title"], row["published_at"], source["id"])
                    if not date or not 6 <= len(title) <= 1500 or datetime.fromisoformat(date) > datetime.fromtimestamp(clock(), timezone.utc):
                        error = "missing_conflicting_or_future_publication_metadata"
                    elif not _within_window(date, bounds):
                        error = "publication_outside_collection_window"
                    else:
                        error = _genre_exclusion(source, title, result.url)
                    if error:
                        state = "quarantine"
                        quarantined += 1
                        db.execute("INSERT OR IGNORE INTO quarantine(url,source_id,title,text,fetched_at,reason,published_at,publication_evidence) "
                                   "VALUES(?,?,?,?,?,?,?,?)", (result.url, source["id"], title, text, _utc(clock()), error, date, canonical(date_evidence)))
                    else:
                        content_hash = hashlib.sha256(" ".join(text.split()).encode()).hexdigest()
                        if db.execute("SELECT 1 FROM articles WHERE content_hash=? OR url=?", (content_hash, result.url)).fetchone():
                            state = "duplicate"
                            duplicate += 1
                        else:
                            signature = _signature(text)
                            title_norm, cluster = _cluster(db, title, signature)
                            article_id = "article_" + content_hash[:32]
                            metadata = {key: source.get(key) for key in ("id", "name", "quality_tier", "topics", "genres", "verification")}
                            metadata["publication_date_evidence"] = date_evidence
                            metadata["publication_date_precision"] = _date_precision(date_evidence)
                            metadata["collection_window"] = {"published_after": published_after, "published_before": published_before,
                                                             "calendar_timezone": "Europe/Moscow", "inclusive": True}
                            db.execute("INSERT INTO articles VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (article_id, result.url, source["id"], title, title_norm,
                                text, content_hash, cluster, canonical(signature), date, _utc(clock()), canonical(metadata)))
                            db.executemany("INSERT INTO anchors VALUES(?,?)", [(anchor, article_id) for anchor in signature[:8]])
                            added += 1
                db.execute("UPDATE queue SET state=?,attempts=attempts+1,error=?,processed_at=? WHERE url=?", (state, error, _utc(clock()), row["url"]))
                source_attempts[sid] = source_attempts.get(sid, 0) + 1
                consecutive_errors[sid] = 0
                db.commit()
            except _Stop as stop:
                db.commit()
                stopped = str(stop)
                break
            except (fetch.CollectionError, ValueError, TypeError, KeyError) as failure:
                errors += 1
                source_attempts[sid] = source_attempts.get(sid, 0) + 1
                db.execute("UPDATE queue SET state='error',attempts=attempts+1,error=?,processed_at=? WHERE url=?",
                           (getattr(failure, "code", type(failure).__name__), _utc(clock()), row["url"]))
                consecutive_errors[sid] = consecutive_errors.get(sid, 0) + 1
                if consecutive_errors[sid] >= 5:
                    _hold_publisher(budget, source, "consecutive_errors", 3600)
                db.commit()
        else:
            stopped = "batch_article_limit" if added >= max_articles else "corpus_goal"
        total = db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        outside_window = sum(not _within_window(row[0], bounds) for row in db.execute("SELECT published_at FROM articles"))
        pending = db.execute("SELECT COUNT(*) FROM queue WHERE state='pending'").fetchone()[0]
        return {"schema_version": CORPUS_SCHEMA, "total": total, "new": added, "errors": errors,
                "quarantined": quarantined, "duplicates": duplicate, "pending": pending,
                "requests": budget.requests, "response_bytes": budget.received, "byte_budget_used": budget.bytes, "stop_reason": stopped,
                "goal": goal, "labels_created": 0,
                "collection_window": {"published_after": published_after, "published_before": published_before,
                                      "calendar_timezone": "Europe/Moscow", "inclusive": True},
                "existing_outside_window": outside_window, "total_in_window": total - outside_window}
