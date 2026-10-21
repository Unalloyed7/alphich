"""Offline deterministic RSS smoke fixture against the REAL PostgreSQL & ingestion worker.

Only callable via CLI inside the container. No HTTP test endpoints, no LLM,
no outbound requests. Bypasses the network exclusively via an injected fetcher.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from urllib.request import Request, urlopen

from sqlalchemy import select

from app.core import Base, make_session_factory
from app.news import Event, Job, SourceDocument
from app.rss import FeedConfig, RssArticle, RssDelivery, collect_once
from app.rss_fetch import Download

HOST = "rss-smoke.alphich-example.org"
FEED_URL = f"https://{HOST}/feed.xml"


class FakeFetcher:
    def __init__(self, article_url: str, phase: str):
        self.article_url = article_url
        self.phase = phase

    def get(self, url, hosts, validators=None):
        if url == FEED_URL:
            if self.phase == "replay":
                if not (validators or {}).get("If-None-Match"):
                    raise RuntimeError("missing_conditional_etag_after_restart")
                return Download(url, 304, {"etag": '"smoke-1"'}, b"")
            feed = (
                '<?xml version="1.0"?><rss version="2.0"><channel><item>'
                '<title>Синтетическая публикация для RSS приёмки</title>'
                f'<link>{self.article_url}</link>'
                '<pubDate>Wed, 21 Oct 2026 09:00:00 +0300</pubDate>'
                '</item></channel></rss>'
            )
            return Download(url, 200, {"etag": '"smoke-1"'}, feed.encode("utf-8"))
        if url == self.article_url and self.phase == "first":
            article = (
                '<html><head><title>Тест RSS</title></head><body><main><article>'
                '<h1>Синтетическая публикация для RSS приёмки</h1>'
                '<p>Это синтетическая тестовая публикация, не представляющая реальную финансовую новость. '
                'Проверяется загрузка RSS, сохранение текста источника и работа очереди событий.</p>'
                '<p>При повторном опросе этой ленты не должны появляться дополнительные события. '
                'После перезапуска коллектора сохранённое состояние должно использоваться повторно.</p>'
                '</article></main></body></html>'
            )
            return Download(url, 200, {"content-type": "text/html; charset=utf-8"}, article.encode("utf-8"))
        raise RuntimeError("unexpected_fixture_request")


def json_api(method: str, path: str, *, token: str | None = None, body: dict | None = None):
    import json as _json
    url = "http://api:8000/api/v1" + path
    headers = {"Accept": "application/json"}
    if token:
        headers["X-Demo-Session"] = token
    payload = None
    if body is not None:
        payload = _json.dumps(body).encode()
        headers["Content-Type"] = "application/json"
    with urlopen(Request(url, data=payload, method=method, headers=headers), timeout=15) as response:
        assert response.status in {200, 201}, f"{path}: HTTP {response.status}"
        return _json.load(response)


def run(test_id: str, phase: str, company: str):
    if len(test_id) != 12 or any(ch not in "0123456789abcdef" for ch in test_id):
        raise ValueError("test_id must be 12 lowercase hex digits")
    feed_id = "rss-smoke-" + test_id
    article_url = f"https://{HOST}/article/{test_id}"
    article_id = hashlib.sha256(article_url.encode()).hexdigest()
    feed = FeedConfig(id=feed_id, url=FEED_URL, article_hosts=[HOST], poll_seconds=900, max_items=1)
    engine, factory = make_session_factory()
    Base.metadata.create_all(engine)  # RSS-only tables are absent on fresh CI databases.
    try:
        report = collect_once(factory, [feed], [company], fetcher=FakeFetcher(article_url, phase),
                              force=True)[0]
        if report["status"] != ("collected" if phase == "first" else "not_modified"):
            raise AssertionError(f"unexpected_poll_result: {report}")
        if report["queued"] != (1 if phase == "first" else 0):
            raise AssertionError(f"duplicate_or_missing_delivery: {report}")
        if report["downloaded"] != (1 if phase == "first" else 0):
            raise AssertionError(f"unexpected_article_fetch: {report}")
        deadline = time.monotonic() + 90
        while True:
            with factory() as db:
                delivery = db.get(RssDelivery, (article_id, company))
                if delivery is None:
                    raise AssertionError("delivery_not_created")
                job = db.get(Job, delivery.job_id)
                source = db.get(SourceDocument, delivery.source_id)
                event = db.scalar(select(Event).where(Event.source_id == source.id))
                count = len(list(db.scalars(select(Job).where(
                    Job.owner_session_hash == hashlib.sha256(("rss-system:" + company).encode()).hexdigest(),
                    Job.source_id == source.id,
                    Job.operation == "publication_ingest"))))
                if count != 1:
                    raise AssertionError(f"duplicate_jobs: {count}")
                if source.kind != "rss_article" or source.fetched_at is None:
                    raise AssertionError("missing_provenance")
                if job.status == "failed":
                    raise AssertionError(f"worker_failed: {job.error}")
                if job.status == "succeeded" and event is not None:
                    event_id, source_id = event.id, source.id
                    break
            if time.monotonic() >= deadline:
                raise AssertionError("worker_timeout")
            time.sleep(1)
        auth = json_api("POST", "/demo-session", body={"company_id": company, "role": "cfo"})
        token = auth["token"]
        try:
            got_event = json_api("GET", "/events/" + event_id, token=token)
            got_source = json_api("GET", "/sources/" + source_id, token=token)
            if got_event.get("id") != event_id or source_id not in got_event.get("source_ids", []):
                raise AssertionError("event_api_provenance_mismatch")
            if got_source.get("id") != source_id or got_source.get("text") != source.body:
                raise AssertionError("source_api_mismatch")
            listing = json_api("GET", f"/companies/{company}/events?limit=50", token=token)
            if not any(x.get("id") == event_id for x in listing.get("items", [])):
                raise AssertionError("event_missing_from_company_list")
        finally:
            token = None
        print(json.dumps({"phase": phase, "status": "passed", "event_id": event_id,
                          "source_id": source_id, "result": report}, ensure_ascii=False))
    finally:
        engine.dispose()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-id", required=True)
    parser.add_argument("--phase", choices=["first", "replay"], required=True)
    parser.add_argument("--company", default="DEMO_A")
    args = parser.parse_args()
    run(args.test_id, args.phase, args.company)


if __name__ == "__main__":
    main()
