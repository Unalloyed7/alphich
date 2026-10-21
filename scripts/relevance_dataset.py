#!/usr/bin/env python3
"""Export frozen, private, leak-resistant training inputs from the local Quality Lab.

This command is READ-ONLY against the review SQLite database. It does not call
an LLM, fetch the internet, train a model or modify the live Alphich system.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from urllib.parse import urlsplit, urlunsplit

SCHEMA_VERSION = "alphich-relevance-dataset-v1"
# Profiles originate from locally verified review_context snapshots. We deliberately
# do not export arbitrary future properties, source rows or private profile notes.
ALLOWED_FEATURES = frozenset({
    "okved_code", "okved_name", "registration_region", "registration_city",
    "product_rko", "product_guarantee", "product_loan", "product_lease",
    "product_deposit", "product_payroll", "product_factoring",
    "fts_flag", "fts_countries",
})


def compact(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(text_or_bytes):
    if isinstance(text_or_bytes, str):
        text_or_bytes = text_or_bytes.encode("utf-8")
    return hashlib.sha256(text_or_bytes).hexdigest()


def read_json(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Expected JSON object: " + str(path))
    return data


def verify_fixtures(folder, versions):
    result = {}
    for version in versions:
        if not re.fullmatch(r"v[0-9]+", version):
            raise ValueError("Unsupported fixture ID " + version)
        path = folder / f"real_news_selection_labels.{version}.json"
        if not path.is_file():
            raise ValueError("Missing fixture: " + str(path))
        raw = path.read_bytes()
        data = json.loads(raw)
        result[version] = {"sha256": digest(raw), "as_of": data.get("as_of"),
                           "version": str(data.get("version") or version),
                           "schema": data.get("rubric_version")}
    return result


def canonical_url(url):
    if not isinstance(url, str):
        return None
    try:
        parts = urlsplit(url)
        if parts.scheme != "https" or not parts.hostname or parts.username or parts.password:
            return None
        # Drop fragment and query trackers, preserve non-tracking query parameters.
        # Important: many CBR URLs identify the article with the `file` query.
        from urllib.parse import parse_qsl, urlencode
        params = sorted((k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
                        if k.lower() not in {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                                              "utm_content", "fbclid", "gclid"})
        path = parts.path or "/"
        return urlunsplit(("https", parts.hostname.lower(), path.rstrip("/") or "/", urlencode(params), ""))
    except ValueError:
        return None


def article_group(row, overrides):
    """Exact text / canonical URL grouping; editorial overrides for syndications.

    Slightly rewritten cross-site reprints WILL NOT be detected automatically.
    """
    if row["event_id"] in overrides:
        return "manual:" + digest(str(overrides[row["event_id"]]))[:20]
    if row["article_text"]:
        return "body:" + digest(" ".join(row["article_text"].split()))[:20]
    url = canonical_url(row["url"])
    return "url:" + digest(url)[:20] if url else "event:" + digest(row["event_id"])[:20]


def profile_from_context(text):
    if text is None:
        return None, None
    context = json.loads(text)
    if not isinstance(context, dict) or not isinstance(context.get("features"), list):
        raise ValueError("Invalid stored company context")
    features = []
    refs = []
    names = set()
    for feature in context["features"]:
        name = feature.get("name") if isinstance(feature, dict) else None
        if name not in ALLOWED_FEATURES:
            # Fail-closed: do not leak unknown private features into ML exports.
            continue
        if name in names:
            raise ValueError("Duplicate feature in verified snapshot: " + name)
        names.add(name)
        features.append({"name": name, "value": feature.get("value")})
        refs.append({"name": name, "source_ref": feature.get("source_ref")})
    features.sort(key=lambda x: x["name"])
    refs.sort(key=lambda x: x["name"])
    if not features:
        return None, None
    return ({"features": features, "snapshot_date": context.get("snapshot_at"),
             "verified_at": context.get("verified_at"),
             "unknown_exposures": ["operational_geography_not_verified",
                                   "financial_contract_terms_unknown"]},
            {"feature_sources": refs, "context_sha256": digest(text)})


def final_grade(db, pair_id):
    a = db.execute("SELECT * FROM adjudications WHERE pair_id=? ORDER BY revision DESC LIMIT 1", (pair_id,)).fetchone()
    if a:
        return int(a["grade"]), "adjudicated", [a["id"]]
    rows = db.execute("""SELECT r.* FROM reviews r JOIN
      (SELECT reviewer, MAX(revision) AS rev FROM reviews WHERE pair_id=? GROUP BY reviewer) x
      ON x.reviewer=r.reviewer AND x.rev=r.revision AND r.pair_id=?
      ORDER BY reviewer""", (pair_id, pair_id)).fetchall()
    if len(rows) != 2:
        return None, "pending_human_review", []
    if any(rows[0][k] != rows[1][k] for k in ("grade", "geography", "scale", "applicability")):
        return None, "requires_adjudication", [r["id"] for r in rows]
    if rows[0]["justification"].strip().casefold() != rows[1]["justification"].strip().casefold():
        # Existing Quality Lab considers this 'agreed'; training needs a human
        # review if the explanations materially differ, so stay conservative.
        return None, "review_justification_difference", [r["id"] for r in rows]
    return int(rows[0]["grade"]), "human_agreement", [r["id"] for r in rows]


def partition(group, seed):
    # Clustered splitting: all companies for an article group stay together.
    fraction = int(digest(f"{seed}|{group}")[:12], 16) / float(16 ** 12)
    return "test" if fraction < .20 else "validation" if fraction < .35 else "train"


def make_rows(db, fixtures, overrides, seed, holdout_company):
    inputs, human, teacher = [], [], []
    records = list(db.execute("SELECT * FROM pairs ORDER BY dataset,event_id,company_id"))
    for row in records:
        dataset = row["dataset"]
        meta = fixtures.get(dataset)
        if meta is None:
            raise ValueError("Unexpected dataset " + dataset)
        if row["fixture_hash"] != meta["sha256"] or row["dataset_version"] != meta["version"]:
            raise ValueError("Source fixture changed since Quality Lab init: " + dataset)
        profile, profile_ref = profile_from_context(row["company_context"])
        body = row["article_text"]
        group = article_group(row, overrides)
        split = "company_holdout" if row["company_id"] == holdout_company else partition(group, seed)
        grade, status, review_ids = final_grade(db, row["id"])
        has_content = bool(body and body.strip())
        has_date = bool(row["published_at"])
        has_profile = profile is not None
        key = digest(compact([dataset, row["event_id"], row["company_id"], meta["sha256"]]))
        inputs.append({
            "schema_version": SCHEMA_VERSION,
            "pair_id": key, "event_id": row["event_id"], "company_id": row["company_id"],
            "article_group_id": group, "split": split,
            "news": {"title": row["title"], "text": body, "published_at": row["published_at"],
                     "source_url": canonical_url(row["url"])},
            "company": profile,
            "versions": {"fixture_dataset": dataset, "fixture_version": meta["version"],
                         "fixture_sha256": meta["sha256"], "fixture_as_of": meta["as_of"],
                         "news_text_sha256": digest(body) if has_content else None,
                         "company_context_sha256": profile_ref["context_sha256"] if profile_ref else None,
                         "schema_version": SCHEMA_VERSION},
            "provenance": {"company_feature_sources": profile_ref["feature_sources"] if profile_ref else []},
            "readiness": {"has_article_text": has_content, "has_published_at": has_date,
                          "has_verified_company_context": has_profile,
                          "eligible_for_training": has_content and has_date and has_profile and grade is not None},
        })
        human.append({"pair_id": key, "grade": grade, "is_relevant": grade >= 4 if grade is not None else None,
                      "status": status, "review_record_ids": review_ids, "split": split})
        teacher_record = json.loads(row["original_prediction_json"])
        teacher.append({"pair_id": key, "grade": teacher_record.get("grade"),
                        "is_relevant": teacher_record.get("selected"),
                        "rationale": teacher_record.get("rationale"),
                        "assessment": teacher_record.get("assessment"),
                        "annotation_type": "preliminary_llm_not_gold", "split": split})
    if holdout_company and not any(r["company_id"] == holdout_company for r in inputs):
        raise ValueError("Company for holdout not found: " + holdout_company)
    if len({r["pair_id"] for r in inputs}) != len(inputs):
        raise ValueError("Duplicate input pair IDs")
    # Known groups and their all-pair partitions must never span article splits.
    article_groups = {}
    for row in inputs:
        if row["split"] == "company_holdout":
            continue
        article_groups.setdefault(row["article_group_id"], set()).add(row["split"])
    if any(len(splits) > 1 for splits in article_groups.values()):
        raise ValueError("Article group leakage across partitions")
    return inputs, human, teacher


def write_jsonl(path, rows):
    with path.open("x", encoding="utf-8") as f:
        os.chmod(path, 0o600)
        for row in rows:
            f.write(compact(row) + "\n")


def export_dataset(db_path, fixture_dir, output_dir, *, teacher_labels=False, seed="alphich-v1",
                   group_overrides=None, heldout_company=None):
    if not db_path.is_file():
        raise ValueError("Missing Quality Lab database: " + str(db_path))
    if output_dir.exists():
        raise ValueError("Output directory already exists (exports are immutable): " + str(output_dir))
    if heldout_company and not isinstance(heldout_company, str):
        raise ValueError("Invalid heldout company ID")
    if not output_dir.parent.is_dir():
        raise ValueError("Create output parent directory first: " + str(output_dir.parent))
    override = read_json(group_overrides) if group_overrides else {}
    if not all(isinstance(k, str) and isinstance(v, str) and v.strip() for k, v in override.items()):
        raise ValueError("Group overrides must map event_id -> non-empty group name")
    with sqlite3.connect(db_path.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.row_factory = sqlite3.Row
        datasets = [x[0] for x in db.execute("SELECT DISTINCT dataset FROM pairs")]
        fixtures = verify_fixtures(fixture_dir, datasets)
        with db:  # Read-only snapshot of the exact DB state.
            db.execute("BEGIN")
            inputs, labels, teachers = make_rows(db, fixtures, override, seed, heldout_company)
            db.rollback()
    temp = Path(tempfile.mkdtemp(prefix=".relevance-export-", dir=output_dir.parent))
    os.chmod(temp, 0o700)
    try:
        write_jsonl(temp / "model_inputs.jsonl", inputs)
        write_jsonl(temp / "human_labels.jsonl", labels)
        if teacher_labels:
            write_jsonl(temp / "teacher_labels_PRELIMINARY.jsonl", teachers)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "data_origin": "independent_local_review_database",
            "export_snapshot_sha256": digest(compact({"inputs": inputs, "human_labels": labels})),
            "seed": seed, "company_holdout": heldout_company,
            "dataset_fixtures": fixtures,
            "article_group_overrides_sha256": digest(group_overrides.read_bytes()) if group_overrides else None,
            "counts": {"total_pairs": len(inputs), "news_text_pairs": sum(r["readiness"]["has_article_text"] for r in inputs),
                       "profile_pairs": sum(r["readiness"]["has_verified_company_context"] for r in inputs),
                       "human_gold_pairs": sum(r["grade"] is not None for r in labels),
                       "training_ready_pairs": sum(r["readiness"]["eligible_for_training"] for r in inputs),
                       "original_llm_positive_pairs": sum(r["grade"] is not None and r["grade"] >= 4 for r in teachers)},
            "splits": {part: sum(row["split"] == part for row in inputs)
                       for part in ("train", "validation", "test", "company_holdout")},
            "warnings": ["Teacher predictions excluded from model_inputs.jsonl to prevent leakage",
                         "Synthetic/reference development labels are not independent human gold",
                         "Cross-domain paraphrased reprints require curated group overrides",
                         "Current model_inputs are private; do not commit to public GitHub"],
            "files": {},
        }
        for filename in ("model_inputs.jsonl", "human_labels.jsonl") + (("teacher_labels_PRELIMINARY.jsonl",) if teacher_labels else ()):
            manifest["files"][filename] = digest((temp / filename).read_bytes())
        target = temp / "manifest.json"
        target.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.chmod(target, 0o600)
        temp.rename(output_dir)
    finally:
        if temp.exists():
            import shutil
            shutil.rmtree(temp)
    return manifest


def main():
    ap = argparse.ArgumentParser(description="Read-only offline export for Alphich relevance ML")
    ap.add_argument("--db", type=Path, default=Path("data/private/quality-review.sqlite3"))
    ap.add_argument("--fixtures-dir", type=Path, default=Path("knowledge/fixtures"))
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--with-teacher-labels", action="store_true", help="Separate preliminary LLM labels for the trainer; NOT model features")
    ap.add_argument("--article-groups", type=Path, help="Private JSON event_id -> reprint group; prevents cross-site leakage")
    ap.add_argument("--heldout-company", help="Do not train on this company; partition its examples separately")
    ap.add_argument("--seed", default="alphich-v1")
    args = ap.parse_args()
    try:
        manifest = export_dataset(args.db, args.fixtures_dir, args.output_dir,
                                  teacher_labels=args.with_teacher_labels, seed=args.seed,
                                  group_overrides=args.article_groups, heldout_company=args.heldout_company)
        print(json.dumps({"output_dir": str(args.output_dir), **manifest["counts"], "splits": manifest["splits"]}, ensure_ascii=False))
    except (OSError, ValueError, sqlite3.DatabaseError, json.JSONDecodeError) as e:
        ap.exit(2, "ERROR: " + str(e) + "\n")


if __name__ == "__main__":
    main()
