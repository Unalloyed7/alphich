"""Small annotated development checks; not a held-out quality benchmark."""

from pathlib import Path

from .common import read_json, read_jsonl, verify_artifacts
from .search import search, select_claims


def evaluate(corpus_dir, cases_path):
    manifest = verify_artifacts(corpus_dir)
    claims = read_jsonl(Path(corpus_dir) / "claims.jsonl")
    results = []
    for case in read_json(cases_path)["cases"]:
        if case["kind"] == "forecast":
            found = select_claims(claims, case["as_of"], case["metric"], case["horizon"])
            actual = [c["annotation_id"] for c in found]
            results.append({"id": case["id"], "kind": "forecast", "passed": set(actual) == set(case["expected_claims"]),
                            "expected": case["expected_claims"], "actual": actual})
        else:
            hits = search(corpus_dir, case["query"], case["as_of"], case.get("limit", 5))
            actual = [(h["document_id"], h["page"]) for h in hits]
            expected = [tuple(pair) for pair in case["expected_pages"]]
            ranks = [i + 1 for i, pair in enumerate(actual) if pair in expected]
            results.append({"id": case["id"], "kind": "search", "passed": bool(ranks) if expected else not actual,
                            "expected": expected, "actual": actual,
                            "recall_at_k": len(set(actual) & set(expected)) / len(set(expected)) if expected else None,
                            "reciprocal_rank": 1 / ranks[0] if ranks else 0 if expected else None})
    positives = [r for r in results if r["kind"] == "search" and r["expected"]]
    return {"corpus_version": manifest["version"], "evaluation_scope": "small_curated_development_set",
            "mode": "lexical_baseline_and_forecast_rules", "cases": len(results),
            "passed": sum(r["passed"] for r in results),
            "mean_recall_at_k": sum(r["recall_at_k"] for r in positives) / len(positives) if positives else None,
            "mrr": sum(r["reciprocal_rank"] for r in positives) / len(positives) if positives else None,
            "results": results}
