#!/usr/bin/env python3
"""Prepare independent review article snapshots via Alphich's bounded public fetcher.

Never calls an LLM, reads teacher labels for decisions, or writes to Quality Lab DB.
Raw fetched text remains a DRAFT until an operator explicitly approves selected IDs.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import re
import sys
import time
from urllib.parse import urlsplit

REPO = Path(__file__).resolve().parent.parent
DEFAULT_PRIVATE = REPO / "data" / "private" / "quality-news"
DATASETS = ("v2", "v3")
# These hosts are from committed review fixtures, NOT an unrestricted URL input.
SOURCE_HOSTS = frozenset({
    "www.cbr.ru", "www.interfax.ru", "stroygaz.ru", "www.cnews.ru",
    "www.agroinvestor.ru", "www.elec.ru", "www.mintrans.gov.ru",
    "mintrans.gov.ru", "zoom.cnews.ru", "www.trendforce.com",
})
TEXT_CAP = 100_000


def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical(obj) -> bytes:
    return (json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_fixtures(fixtures: Path, dataset: str) -> tuple[list[dict], dict]:
    if dataset not in DATASETS:
        raise ValueError("dataset must be v2 or v3")
    path = fixtures / f"real_news_selection_labels.{dataset}.json"
    raw = path.read_bytes()
    data = json.loads(raw)
    sources = data.get("sources")
    labels = data.get("labels")
    if not isinstance(sources, list) or not isinstance(labels, list):
        raise ValueError("invalid fixture structure")
    ids = {s.get("event_id") for s in sources if isinstance(s, dict)}
    if len(ids) != len(sources) or None in ids:
        raise ValueError("duplicate or missing event_id")
    for row in labels:
        if row.get("event_id") not in ids:
            raise ValueError("orphan label")
    safe_sources = []
    for s in sources:
        url = s.get("url")
        if not isinstance(url, str):
            raise ValueError("missing source URL")
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.hostname not in SOURCE_HOSTS
                or parts.username is not None or parts.password is not None
                or parts.port not in (None, 443)):
            raise ValueError("source URL outside pinned HTTPS hosts")
        title = s.get("title") or s.get("short_title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("missing title")
        # Construct a *fresh* allowlist. Do not include original AI grades or reasons.
        safe_sources.append({"event_id": s["event_id"], "title": title, "url": url})
    return safe_sources, {"fixture_sha256": digest(raw), "fixture_version": data.get("version"),
                          "dataset": dataset, "source_count": len(safe_sources)}


def article_date_valid(value) -> bool:
    if value is None:
        return True
    if not isinstance(value, str) or len(value) > 40:
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
        return True
    except ValueError:
        return False


def capture_one(source: dict, *, get, extract, get_date) -> tuple[dict, dict]:
    """Deps injected for offline tests; production invokes hardened Alphich code."""
    event_id = source["event_id"]
    origin = source["url"]
    result = get(origin, frozenset({urlsplit(origin).hostname}))
    if result.status != 200:
        raise ValueError(f"unexpected_http_{result.status}")
    # All redirects are restricted to the EXACT requested hostname by the fetcher.
    final = urlsplit(result.url)
    if final.scheme != "https" or final.hostname != urlsplit(origin).hostname:
        raise ValueError("unsafe_final_url")
    text = extract(result.body, result.url, result.headers.get("content-type", ""))
    if not isinstance(text, str) or not 80 <= len(text) <= TEXT_CAP:
        raise ValueError("invalid_extracted_text")
    value = get_date(result.body)
    if not article_date_valid(value):
        raise ValueError("invalid_publication_date")
    article = {"text": text.strip(), "published_at": value}
    info = {"event_id": event_id, "title": source["title"], "original_url": origin,
            "final_url": result.url, "text_sha256": digest(article["text"].encode("utf-8")),
            "text_chars": len(article["text"]), "published_at": value,
            "publication_date_provenance": "html_article_published_time" if value else "not_found",
            "review_status": "downloaded_not_human_verified"}
    return article, info


def preview_html(draft: dict, records: list[dict]) -> str:
    """Standalone offline view; escaped content cannot execute article HTML."""
    e = html.escape
    parts = [
        '<!doctype html><html lang="ru"><head><meta charset="utf-8"><title>Alphich draft review</title>',
        '<style>body{font:16px/1.5 system-ui;max-width:900px;margin:30px auto;padding:20px;background:#f6f7f9}',
        'article{background:white;margin:20px 0;padding:25px;border:1px solid #d4dbe5;border-radius:10px}',
        'pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.5 system-ui}',
        '.warning{background:#fff1d4;padding:12px}</style></head><body>',
        '<h1>Черновики статей для независимой проверки</h1>',
        '<p>Скачанный текст еще не является подтвержденным: сверьте его с источником.</p>',
    ]
    for entry in records:
        if entry["state"] != "draft":
            continue
        event_id = entry["event_id"]
        date_text = entry["published_at"] or "Дата публикации не подтверждена в metadata"
        parts.extend([
            '<article><h2>' + e(entry["title"]) + '</h2>',
            '<p><code>' + e(event_id) + '</code> · ' + e(date_text) + '</p>',
            '<p><a target="_blank" rel="noopener noreferrer" href="' + e(entry["original_url"], quote=True) + '">Сверить оригинал ↗</a></p>',
            '<p class="warning">НЕ ПРОВЕРЕНО ЧЕЛОВЕКОМ. Сверьте полноту, дату и соответствие тексту.</p>',
            '<pre>' + e(draft[event_id]["text"]) + '</pre></article>',
        ])
    parts.append('</body></html>')
    return '\n'.join(parts)



def fetch_batch(*, fixtures: Path, private_root: Path, dataset: str, batch: str,
                limit: int, delay: float, get, extract, get_date, sleep=time.sleep):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}", batch):
        raise ValueError("invalid batch ID")
    if not 1 <= limit <= 62 or delay < 0.5:
        raise ValueError("limit must be 1..62 and delay >= 0.5 seconds")
    sources, meta = load_fixtures(fixtures, dataset)
    root = private_root / batch
    root.mkdir(parents=True, exist_ok=False)  # never overwrite an earlier archive
    root.chmod(0o700)
    draft, records, failed = {}, [], 0
    try:
        for index, source in enumerate(sources[:limit]):
            if index:
                sleep(delay)
            try:
                article, info = capture_one(source, get=get, extract=extract, get_date=get_date)
                draft[source["event_id"]] = article
                records.append({"state": "draft", **info})
            except Exception as exc:
                # No exception bodies or downloaded response bytes in diagnostic logs.
                code = getattr(exc, "code", None)
                if not isinstance(code, str) or not re.fullmatch(r"[a-z0-9_]{2,70}", code):
                    code = "fetch_or_extract_failed"
                records.append({"event_id": source["event_id"], "original_url": source["url"],
                                "state": "failed", "error": code})
                failed += 1
        manifest = {"schema_version": "alphich-news-capture-v1", **meta,
                    "retrieved_at": utcnow(), "requested": min(limit, len(sources)),
                    "downloaded": len(draft), "failed": failed, "status": "unreviewed_draft",
                    "records": records}
        for name, obj in [("draft_articles.json", draft), ("manifest.json", manifest)]:
            path = root / name
            with path.open("xb") as out:
                out.write(canonical(obj))
            path.chmod(0o600)
        preview = root / "preview.html"
        with preview.open("xb") as out:
            out.write(preview_html(draft, records).encode("utf-8"))
        preview.chmod(0o600)
        return {"batch_dir": str(root), "downloaded": len(draft), "failed": failed,
                "requested": min(limit, len(sources))}
    except BaseException:
        # Leave any partial data in a private path for explicit operator inspection.
        raise


def approve(*, batch_dir: Path, event_ids: list[str]) -> dict:
    """Explicit human approval, no overwrite; approval cannot bypass SHA validation."""
    root = batch_dir
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    draft = json.loads((root / "draft_articles.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "unreviewed_draft":
        raise ValueError("invalid manifest")
    if not event_ids or len(event_ids) != len(set(event_ids)):
        raise ValueError("supply distinct reviewed event IDs")
    by_id = {r["event_id"]: r for r in manifest["records"] if r["state"] == "draft"}
    out = {}
    for event in event_ids:
        if event not in by_id or event not in draft:
            raise ValueError("requested ID has no downloadable draft: " + event)
        row = draft[event]
        if digest(row["text"].encode("utf-8")) != by_id[event]["text_sha256"]:
            raise ValueError("draft content changed since capture: " + event)
        if row.get("published_at") != by_id[event]["published_at"]:
            raise ValueError("publication metadata changed since capture: " + event)
        out[event] = {"text": row["text"], "published_at": row["published_at"]}
    path = root / "review_articles_APPROVED.json"
    with path.open("xb") as file:
        file.write(canonical(out))
    path.chmod(0o600)
    return {"approved_articles": len(out), "output": str(path)}


def main():
    parser = argparse.ArgumentParser(description="Local bounded public article snapshots for Quality Lab; no LLM")
    sub = parser.add_subparsers(dest="action", required=True)
    fetch = sub.add_parser("fetch", help="EXPLICIT network fetch, unreviewed private draft")
    fetch.add_argument("--fixtures-dir", type=Path, default=Path("knowledge/fixtures"))
    fetch.add_argument("--dataset", choices=DATASETS, default="v3")
    fetch.add_argument("--batch", required=True)
    fetch.add_argument("--limit", type=int, default=5)
    fetch.add_argument("--delay", type=float, default=1.0)
    fetch.add_argument("--fetch", action="store_true", required=True,
                       help="explicit permission to access public news URLs; no LLM calls")
    check = sub.add_parser("approve", help="approve articles manually inspected against originals")
    check.add_argument("--batch", required=True)
    check.add_argument("--ids", nargs="+", required=True)
    args = parser.parse_args()
    try:
        if args.action == "fetch":
            # Imported only at execution; offline unit tests need no backend packages.
            sys.path.insert(0, str(REPO / "backend"))
            from app.rss_fetch import PublicFetcher
            from app.rss_parse import article_text, article_publication_date
            value = fetch_batch(fixtures=args.fixtures_dir, private_root=DEFAULT_PRIVATE,
                                dataset=args.dataset, batch=args.batch, limit=args.limit,
                                delay=args.delay, get=PublicFetcher().get, extract=article_text,
                                get_date=article_publication_date)
            print(json.dumps(value, ensure_ascii=False))
            print("DRAFT ONLY: inspect each source before 'approve' and Quality Lab 'enrich'.")
        else:
            if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,49}", args.batch):
                raise ValueError("invalid batch ID")
            print(json.dumps(approve(batch_dir=DEFAULT_PRIVATE / args.batch, event_ids=args.ids), ensure_ascii=False))
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        parser.exit(2, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()
