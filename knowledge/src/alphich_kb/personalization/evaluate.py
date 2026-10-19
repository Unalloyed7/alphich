"""Open fictional relevance matrix; exercises selection, not real-world quality."""

import math
import statistics

from ..common import KnowledgeError, file_hash, read_json
from .cli import load_rows
from .inputs import SelectionScope
from .selector import NewsSelector


def ndcg(grades, ideal, k=5):
    dcg = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(grades[:k]))
    best = sum((2 ** grade - 1) / math.log2(rank + 2) for rank, grade in enumerate(sorted(ideal, reverse=True)[:k]))
    return dcg / best if best else None


def evaluate_selection(profiles_path, events_path, labels_path, encoder=None):
    profiles = {p.get("company_id", p.get("id")): p for p in load_rows(profiles_path, "companies")}
    events = load_rows(events_path, "events")
    labels = read_json(labels_path)
    grades = {(row["company_id"], row["event_id"]): row["grade"] for row in labels["labels"]}
    if len(grades) != len(labels["labels"]):
        raise KnowledgeError("Duplicate company/event grade")
    selector = NewsSelector(events, encoder)
    reports = {}
    for mode in (["lexical", "dense", "hybrid"] if encoder else ["lexical"]):
        results = []
        for company_id in labels["company_ids"]:
            if encoder is not None and hasattr(encoder, "_cached_query"):
                encoder._cached_query.cache_clear()
            scope = SelectionScope(company_id, labels["as_of"], frozenset(e["id"] for e in events))
            result = selector.select(profiles[company_id], scope, mode=mode)
            if result["mode"] != mode:
                raise KnowledgeError("Evaluation cannot substitute lexical fallback for dense/hybrid")
            selected = [{"event_id": item["event_id"], "priority": item["priority"], "grade": grades[(company_id, item["event_id"])],
                         "tier": item["relevance_tier"], "status": item["status"], "reason_codes": item["reason_codes"]}
                        for item in result["items"]]
            ideal = [grade for (uid, _), grade in grades.items() if uid == company_id]
            results.append({"company_id": company_id, "ndcg_at_5": ndcg([s["grade"] for s in selected], ideal),
                            "unsupported_items": sum(item["grade"] == 0 for item in selected), "items": selected,
                            "extraction_candidates": [item["event_id"] for item in result["extraction_candidates"]],
                            "elapsed_ms": result["trace"]["elapsed_ms"], "retrieval_run_id": result["retrieval_run_id"]})
        reports[mode] = {"mean_ndcg_at_5": statistics.mean(r["ndcg_at_5"] for r in results if r["ndcg_at_5"] is not None),
                         "unsupported_items": sum(r["unsupported_items"] for r in results),
                         "p50_ms": statistics.median(r["elapsed_ms"] for r in results), "results": results}
    return {"scope": "fictional_open_development_matrix_not_real_news_quality", "label_pairs": len(grades),
            "profiles_sha256": file_hash(profiles_path), "events_sha256": file_hash(events_path),
            "labels_sha256": file_hash(labels_path), "as_of": labels["as_of"],
            "embedding_space_id": encoder.space_id if encoder else None,
            "timing": "one pass per mode, warm worker, query cache cleared per company; no generation calls",
            "modes": reports}
