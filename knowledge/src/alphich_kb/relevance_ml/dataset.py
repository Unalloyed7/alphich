"""Versioned, source-bound pair contracts, isolated from teacher explanations.

References describe supplied evidence; validation does not independently verify it.
No fetching, generation, company lookup or label inference happens here.
"""
from __future__ import annotations

import copy
import json
import math
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from ..common import KnowledgeError, canonical, digest
from . import DATASET_SCHEMA, PAIR_SCHEMA

FACT_NAMES = frozenset({
    "industry", "okved_code", "okved_name", "registration_region", "operating_regions",
    "products", "loan", "deposit", "lease", "guarantee", "factoring", "rko", "payroll",
    "fts", "fts_flag", "fts_countries", "revenue_raw", "revenue_unit", "revenue_currency",
    "revenue_period", "revenue_scope", "employees_band", "business_goals",
    "organization_type", "ownership_form", "company_age_band", "business_stage_label",
    "negative_mark_raw", "employees_fns_band", "employees_spark_band", "city_registration", "client_type",
})
PRODUCT_NAMES = frozenset({"loan", "deposit", "lease", "guarantee", "factoring", "rko", "payroll"})


def _fact_value(name, value):
    def string_or_null(item):
        return item is None or isinstance(item, str)
    def strings(item):
        return isinstance(item, list) and len(item) <= 100 and all(isinstance(s, str) for s in item)
    valid = False
    if name == "industry":
        valid = isinstance(value, dict) and set(value) <= {"code", "name"} and all(string_or_null(v) for v in value.values())
    elif name == "products":
        valid = (isinstance(value, dict) and set(value) <= PRODUCT_NAMES and
                 all(v is None or type(v) is bool for v in value.values()))
    elif name == "fts":
        valid = (isinstance(value, dict) and set(value) <= {"flag", "countries"} and
                 (value.get("flag") is None or type(value["flag"]) is bool) and strings(value.get("countries", [])))
    elif name in PRODUCT_NAMES | {"fts_flag"}:
        valid = value is None or type(value) is bool
    elif name in {"operating_regions", "fts_countries"}:
        valid = value is None or strings(value)
    elif name == "business_goals":
        valid = string_or_null(value) or strings(value)
    elif name == "revenue_raw":
        valid = value is None or isinstance(value, (str, int, float)) and not isinstance(value, bool)
    else:
        valid = string_or_null(value)
    if not valid:
        raise KnowledgeError(f"Invalid source-bound fact value: {name}")


def text(value, name, maximum=200000):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise KnowledgeError(f"{name}: expected nonempty text, maximum {maximum} characters")
    return value.strip()


def instant(value):
    try:
        dt = datetime.fromisoformat(text(value, "timestamp", 80).replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError()
        return dt.astimezone(timezone.utc)
    except (ValueError, TypeError, OverflowError):
        raise KnowledgeError("Timestamp must be an ISO-8601 instant with an explicit timezone") from None


def _finite(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise KnowledgeError("Nonfinite JSON value")
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise KnowledgeError("JSON object keys must be strings")
        for v in value.values():
            _finite(v)
    elif isinstance(value, list):
        for v in value:
            _finite(v)
    elif value is not None and not isinstance(value, (str, bool, int, float)):
        raise KnowledgeError("Not a JSON value")


def input_payload(row):
    """Copy only prediction-time fields; labels and arbitrary extras never enter prompts/features."""
    return {"schema_version": PAIR_SCHEMA, "company": copy.deepcopy(row["company"]),
            "article": copy.deepcopy(row["article"]), "as_of": row["as_of"], "role": row["role"]}


def input_hash(row):
    return digest(input_payload(row))


def validate_pair(row):
    if not isinstance(row, dict) or row.get("schema_version", PAIR_SCHEMA) != PAIR_SCHEMA:
        raise KnowledgeError("Unsupported relevance pair schema")
    _finite(row)
    company, article = row.get("company"), row.get("article")
    if not isinstance(company, dict) or not isinstance(article, dict):
        raise KnowledgeError("Pair requires company and article objects")
    facts = company.get("facts")
    if not isinstance(facts, list) or not facts or len(facts) > 64:
        raise KnowledgeError("Company requires 1..64 source-bound facts")
    normalized_facts = []
    ids, names = set(), set()
    for fact in facts:
        if not isinstance(fact, dict) or fact.get("name") not in FACT_NAMES or "value" not in fact:
            raise KnowledgeError("Unsupported company fact; labels/reasons are not company evidence")
        fid = text(fact.get("id"), "fact.id", 300)
        if fid in ids or fact["name"] in names:
            raise KnowledgeError("Company fact IDs and names must be unique")
        ids.add(fid); names.add(fact["name"])
        value = fact["value"]
        _fact_value(fact["name"], value)
        if len(canonical(value)) > 12000:
            raise KnowledgeError("Company fact value too large")
        normalized_facts.append({"id": fid, "name": fact["name"], "value": copy.deepcopy(value),
                                 "source_ref": text(fact.get("source_ref"), "fact.source_ref", 4000)})
    if sum(len(canonical(f)) for f in normalized_facts) > 32000:
        raise KnowledgeError("Company fact snapshot exceeds 32000 characters")
    normalized_company = {"company_id": text(company.get("company_id"), "company_id", 160),
                          "profile_version": text(company.get("profile_version"), "profile_version", 300),
                          "facts": sorted(normalized_facts, key=lambda f: f["name"])}
    url = text(article.get("url"), "article.url", 8000)
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
        raise KnowledgeError("Article URL requires HTTP(S), a host and no credentials")
    as_of = instant(row.get("as_of")).isoformat()
    published = article.get("published_at")
    if published is not None:
        published = instant(published).isoformat()
        if instant(published) > instant(as_of):
            raise KnowledgeError("Article was published after the evaluation instant")
    normalized_article = {"article_id": text(article.get("article_id"), "article_id", 300),
                          "source_version": text(article.get("source_version"), "source_version", 300),
                          "title": text(article.get("title"), "article.title", 1500),
                          "text": text(article.get("text"), "article.text"), "url": url,
                          "published_at": published}
    normalized_article["cluster_id"] = text(article.get("cluster_id") or
        "content_" + digest(normalized_article["text"]), "article.cluster_id", 300)
    role = row.get("role", "cfo")
    if role not in {"cfo", "ceo"}:
        raise KnowledgeError("Role must be cfo or ceo")
    result = {"schema_version": PAIR_SCHEMA, "company": normalized_company, "article": normalized_article,
              "as_of": as_of, "role": role}
    key = "pair_" + input_hash(result)
    if row.get("pair_id", key) != key:
        raise KnowledgeError("pair_id does not match the immutable input snapshot")
    result["pair_id"] = key
    label = row.get("label")
    if label is not None:
        if not isinstance(label, dict) or type(label.get("grade")) is not int or not 0 <= label["grade"] <= 5:
            raise KnowledgeError("Label grade must be an integer 0..5")
        if label.get("origin") not in {"human", "teacher"}:
            raise KnowledgeError("Label origin must be human or teacher")
        result["label"] = copy.deepcopy(label)
        # A supplied version cannot silently bind a label to a different input.
        if label.get("input_hash", input_hash(result)) != input_hash(result):
            raise KnowledgeError("Label is bound to another company/article snapshot")
    return result


DATASET_MAX_BYTES = 512 * 1024 * 1024
JSON_DOCUMENT_MAX_BYTES = 128 * 1024 * 1024
JSONL_LINE_MAX_BYTES = 1024 * 1024


def _strict_json(content):
    def unique(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj:
                raise KnowledgeError("Duplicate JSON key")
            obj[key] = value
        return obj
    def reject(_):
        raise KnowledgeError("Nonfinite JSON number")
    try:
        return json.loads(content, object_pairs_hook=unique, parse_constant=reject)
    except (ValueError, TypeError, RecursionError) as exc:
        if isinstance(exc, KnowledgeError):
            raise
        raise KnowledgeError("Invalid relevance dataset JSON") from None


def _io_limits(limit, max_bytes, max_line_bytes):
    if limit is not None and (type(limit) is not int or limit < 1):
        raise KnowledgeError("Dataset prefix limit must be a positive integer")
    if type(max_bytes) is not int or not 1 <= max_bytes <= 1024 * 1024 * 1024:
        raise KnowledgeError("Dataset byte limit must be 1 byte..1 GiB")
    if type(max_line_bytes) is not int or not 1 <= max_line_bytes <= JSONL_LINE_MAX_BYTES:
        raise KnowledgeError("JSONL line byte limit must be 1 byte..1 MiB")


def iter_dataset(path, limit=None, *, max_bytes=DATASET_MAX_BYTES, max_line_bytes=JSONL_LINE_MAX_BYTES):
    """Yield validated rows, retaining only visited pair IDs for duplicate detection.

    JSONL is streamed with a bounded binary readline. A requested prefix does
    not read or validate the suffix; callers must not describe it as a full-file
    audit. Legacy JSON envelopes/arrays remain supported up to 128 MiB because
    the stdlib JSON parser materializes a complete document. Use JSONL for large
    corpora. Total file size defaults to 512 MiB and is explicitly bounded at 1 GiB.
    """
    _io_limits(limit, max_bytes, max_line_bytes)
    path = Path(path)
    if not path.is_file() or path.stat().st_size > max_bytes:
        raise KnowledgeError("Dataset is missing or exceeds the configured byte limit")
    seen, visited = set(), 0

    def checked(value):
        try:
            row = validate_pair(value)
        except (ValueError, TypeError, RecursionError) as error:
            if isinstance(error, KnowledgeError):
                raise
            raise KnowledgeError("Invalid relevance pair JSON") from None
        if row["pair_id"] in seen:
            raise KnowledgeError("Duplicate company/article snapshot pairs")
        seen.add(row["pair_id"])
        return row

    if path.suffix.casefold() == ".jsonl":
        total = 0
        with path.open("rb") as stream:
            while limit is None or visited < limit:
                line = stream.readline(max_line_bytes + 1)
                if not line:
                    break
                total += len(line)
                if total > max_bytes:
                    raise KnowledgeError("Dataset grew beyond the configured byte limit")
                if len(line) > max_line_bytes:
                    raise KnowledgeError("JSONL row exceeds the configured line byte limit")
                if not line.strip():
                    continue
                row = checked(_strict_json(line))
                visited += 1
                yield row
    else:
        if path.stat().st_size > min(max_bytes, JSON_DOCUMENT_MAX_BYTES):
            raise KnowledgeError("Legacy JSON document exceeds 128 MiB; use streaming JSONL")
        # read() is bounded even if the file grows after stat().
        with path.open("rb") as stream:
            content = stream.read(min(max_bytes, JSON_DOCUMENT_MAX_BYTES) + 1)
        if len(content) > min(max_bytes, JSON_DOCUMENT_MAX_BYTES):
            raise KnowledgeError("Legacy JSON document exceeds its configured byte limit; use JSONL")
        value = _strict_json(content)
        if isinstance(value, dict):
            if value.get("schema_version") != DATASET_SCHEMA:
                raise KnowledgeError("Unsupported relevance dataset schema")
            values = value.get("rows")
        else:
            values = value
        if not isinstance(values, list):
            raise KnowledgeError("Dataset rows must be a list")
        for value in values:
            if limit is not None and visited >= limit:
                break
            row = checked(value)
            visited += 1
            yield row
    if not visited:
        raise KnowledgeError("Dataset requires at least one pair")


def read_dataset(path, limit=None, *, max_bytes=DATASET_MAX_BYTES, max_line_bytes=JSONL_LINE_MAX_BYTES):
    """Read validated rows; prefer iter_dataset when a full in-memory list is unnecessary."""
    return list(iter_dataset(path, limit=limit, max_bytes=max_bytes, max_line_bytes=max_line_bytes))


def write_private_json(path, value):
    """Create-only private artifact. Never overwrite reviewed labels or model versions."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    _finite(value)
    raw = (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode()
    fd, stage = tempfile.mkstemp(prefix=".relevance-", dir=destination.parent)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "wb") as out:
            out.write(raw)
        # link is an atomic create-if-absent on the same filesystem.
        os.link(stage, destination)
    finally:
        os.unlink(stage)


@contextmanager
def _create_private_stream(path):
    """Atomic create-if-absent output: validation failures never leave a partial dataset."""
    destination = Path(path)
    if os.path.lexists(destination):
        raise FileExistsError(str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, stage = tempfile.mkstemp(prefix=".relevance-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as out:
            os.fchmod(out.fileno(), 0o600)
            yield out
        os.link(stage, destination)
    finally:
        os.unlink(stage)


def write_dataset(path, rows, *, max_bytes=DATASET_MAX_BYTES, **metadata):
    """Stream a private create-only dataset without constructing a giant JSON string.

    JSONL stores canonical pair rows only; callers keep metadata/reports in a
    separate explicit artifact. Legacy JSON keeps its original versioned envelope
    and metadata. Both formats reject duplicate snapshots before publication.
    """
    _io_limits(None, max_bytes, JSONL_LINE_MAX_BYTES)
    _finite(metadata)
    jsonl = Path(path).suffix.casefold() == ".jsonl"
    byte_limit = max_bytes if jsonl else min(max_bytes, JSON_DOCUMENT_MAX_BYTES)
    seen, count, total = set(), 0, 0
    with _create_private_stream(path) as out:
        def emit(value):
            nonlocal total
            total += len(value)
            if total > byte_limit:
                raise KnowledgeError("Dataset exceeds its configured byte limit; use JSONL for large corpora")
            out.write(value)

        if not jsonl:
            emit(("{\"schema_version\":" + json.dumps(DATASET_SCHEMA) + ",\"metadata\":"
                  + json.dumps(metadata, ensure_ascii=False, allow_nan=False) + ",\"rows\":[").encode("utf-8"))
        for source in rows:
            row = validate_pair(source)
            if row["pair_id"] in seen:
                raise KnowledgeError("Duplicate company/article snapshot pairs")
            seen.add(row["pair_id"])
            encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False)).encode("utf-8")
            if jsonl:
                if len(encoded) + 1 > JSONL_LINE_MAX_BYTES:
                    raise KnowledgeError("JSONL row exceeds 1 MiB")
                emit(encoded + b"\n")
            else:
                emit((b"," if count else b"") + encoded)
            count += 1
        if not jsonl:
            emit(b"]}\n")
    return {"format": "jsonl" if jsonl else "json", "rows": count, "bytes": total,
            "metadata_included": not jsonl}


def dataset_summary(rows):
    labels = [r["label"] for r in rows if r.get("label") is not None]
    return {"pairs": len(rows), "articles": len({r["article"]["article_id"] for r in rows}),
            "companies": len({r["company"]["company_id"] for r in rows}),
            "labeled": len(labels), "positive_4_5": sum(l["grade"] >= 4 for l in labels),
            "teacher_labels": sum(l["origin"] == "teacher" for l in labels),
            "human_labels": sum(l["origin"] == "human" for l in labels),
            "missing_publication_date": sum(r["article"]["published_at"] is None for r in rows),
            "input_fingerprint": digest([input_hash(r) for r in rows])}


def import_fixtures(fixtures, events, profiles):
    """Join provisional labels to independently captured text/profile snapshots.

    Missing snapshots are quarantined, never reconstructed from teacher rationale.
    Supports current v2/v3 fixtures and existing selection-real-news captures.
    """
    from ..personalization.profiles import normalize_company
    event_rows = events.get("events", [])
    by_id = {str(r["id"]): r for r in event_rows}
    by_url = {s.get("url"): r for r in event_rows for s in r.get("sources", [])}
    companies = {r["company_id"]: r for r in profiles.get("companies", [])}
    rows, quarantine = [], []
    for fixture in fixtures:
        for source in fixture.get("sources", []):
            event = by_id.get(source["event_id"]) or by_url.get(source["url"])
            for cid, grade in source.get("grades", {}).items():
                reason = None
                if event is None:
                    reason = "article_snapshot_missing"
                elif cid not in companies:
                    reason = "company_snapshot_missing"
                if reason:
                    quarantine.append({"article_id": source["event_id"], "company_id": cid, "reason": reason})
                    continue
                company = companies[cid]
                normalized = normalize_company(company)
                fact_rows = []
                for feature in company.get("features", []):
                    if feature["name"] not in FACT_NAMES or feature["id"] not in normalized["features"]:
                        continue
                    refs = feature.get("source_refs")
                    if not refs:
                        continue
                    fact_rows.append({"id": feature["id"], "name": feature["name"], "value": feature["value"],
                                      "source_ref": canonical(refs)})
                snapshots = [s for s in event.get("sources", [])
                             if s.get("url") == source["url"] and s.get("text") and
                             s.get("completeness") in {"extracted_article_text", "full_text"}]
                if len(snapshots) != 1 or not fact_rows:
                    quarantine.append({"article_id": source["event_id"], "company_id": cid,
                                       "reason": "full_text_or_source_bound_facts_missing"})
                    continue
                snapshot = snapshots[0]
                row = {"company": {"company_id": cid, "profile_version": company["profile_version"], "facts": fact_rows},
                       "article": {"article_id": source["event_id"], "source_version": str(snapshot.get("version", "1")) +
                                   ":" + digest(snapshot["text"]), "title": snapshot["title"], "text": snapshot["text"],
                                   "url": snapshot["url"], "published_at": snapshot.get("published_at"),
                                   "cluster_id": "article_" + digest(snapshot["text"])},
                       "as_of": fixture["as_of"], "role": "cfo"}
                try:
                    row = validate_pair(row)
                    row["label"] = {"grade": grade, "origin": "teacher", "labeler": fixture.get("annotation_method", "agent"),
                                    "label_version": fixture["version"], "input_hash": input_hash(row),
                                    "reason": source.get("rationale", {}).get(cid, ""),
                                    "independently_reviewed": False, "scope": "historical_agent_annotation"}
                    rows.append(validate_pair(row))
                except KnowledgeError as error:
                    quarantine.append({"article_id": source["event_id"], "company_id": cid, "reason": str(error)})
    unique = {}
    for row in rows:
        if row["pair_id"] in unique:
            quarantine.append({"pair_id": row["pair_id"], "reason": "duplicate_pair"})
        else:
            unique[row["pair_id"]] = row
    return list(unique.values()), quarantine
