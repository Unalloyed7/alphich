"""Offline lexical baseline and deterministic selection of reviewed forecasts."""

import re
import sqlite3
from datetime import date
from pathlib import Path

from .common import KnowledgeError, read_jsonl, verify_artifacts


def available(item, as_of):
    date.fromisoformat(as_of)
    published = item.get("published_at")
    return published is not None and date.fromisoformat(published) <= date.fromisoformat(as_of)


class LexicalIndex:
    """One request-local index of already eligible items, reused by all facets."""

    def __init__(self, selected):
        self.db = sqlite3.connect(":memory:")
        try:
            self.db.execute("CREATE VIRTUAL TABLE fragments USING fts5(title, body, tokenize='unicode61')")
            self.db.executemany("INSERT INTO fragments(rowid, title, body) VALUES (?, ?, ?)",
                               ((i + 1, c["title"], c["text"]) for i, c in enumerate(selected)))
        except BaseException:
            self.db.close()
            raise

    def search(self, query):
        terms = re.findall(r"[^\W_]+", query.lower().replace("ё", "е"), re.UNICODE)
        if not terms:
            return []
        # FTS5 prefix matching is an explicit baseline, not a morphological model.
        expression = " OR ".join(f'"{term}"*' for term in dict.fromkeys(terms))
        rows = self.db.execute(
            "SELECT rowid, bm25(fragments, 0.2, 1.0) FROM fragments WHERE fragments MATCH ? ORDER BY 2, rowid",
            (expression,)).fetchall()
        return [(rowid - 1, score) for rowid, score in rows]

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.db.close()


def lexical_candidates(selected, query, *, index=None):
    """Rank exactly the supplied eligible items, before any top-K truncation."""
    if index is not None:
        return index.search(query)
    with LexicalIndex(selected) as prepared:
        return prepared.search(query)


def search(corpus_dir, query, as_of, limit=5, topics=None):
    date.fromisoformat(as_of)
    if not 1 <= limit <= 50:
        raise KnowledgeError("Search limit must be 1..50")
    verify_artifacts(corpus_dir)
    chunks = read_jsonl(Path(corpus_dir) / "chunks.jsonl")
    selected = [c for c in chunks if available(c, as_of) and (not topics or set(topics).intersection(c["topics"]))]
    results, seen_pages = [], set()
    rows = lexical_candidates(selected, query)
    for rowid, score in rows:
        chunk = selected[rowid]
        if chunk["parent_id"] in seen_pages:
            continue
        seen_pages.add(chunk["parent_id"])
        results.append({"evidence_id": chunk["id"], "document_id": chunk["document_id"],
                        "parent_id": chunk["parent_id"], "page": chunk["page"], "title": chunk["title"],
                        "published_at": chunk["published_at"], "url": chunk["url"], "text": chunk["text"],
                        "ocr_checked": chunk["ocr_checked"], "quality_flags": chunk["quality_flags"],
                        "mode": "lexical_baseline", "bm25": score})
        if len(results) == limit:
            break
    return results


def select_claims(claims, as_of, metric, horizon, attributed_to="alfa_bank", geography="RU", scenario="base"):
    """Selection is scoped by author, horizon and role before applying revisions."""
    date.fromisoformat(as_of)
    candidates = [c for c in claims if available(c, as_of)
                  and c["verification_status"] == "visually_verified"
                  and c["kind"] == "forecast" and c["position_role"] == "current_in_document"
                  and c["metric"] == metric and c["forecast_horizon"] == horizon
                  and c["attributed_to"] == attributed_to and c["geography"] == geography and c["scenario"] == scenario]
    replaced = {c["revision_of"] for c in candidates if c["revision_of"]}
    active = [c for c in candidates if c["id"] not in replaced]
    return sorted(active, key=lambda c: (c["published_at"], c["id"]), reverse=True)
