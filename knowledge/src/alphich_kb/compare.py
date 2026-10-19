"""Same-corpus retrieval comparison with page labels and temporal/access checks."""

import math
import platform
import resource
import sys
from pathlib import Path

import numpy as np

from .common import file_hash, read_json
from .retriever import BankRetriever, RetrievalScope


def compare_retrieval(index_dir, cases_path, encoder):
    retriever = BankRetriever(index_dir, encoder)
    cases = read_json(cases_path)["cases"]
    reports = {}
    for mode in ("lexical", "dense", "hybrid"):
        results = []
        for case in cases:
            # Measure warm-model, uncached queries equally for dense and hybrid.
            if hasattr(encoder, "_cached_query"):
                encoder._cached_query.cache_clear()
            scope = RetrievalScope(case["as_of"], frozenset(case.get("allowed_documents", retriever.documents.keys())))
            result = retriever.retrieve(case["query"], scope, mode=mode, limit=case.get("limit", 5),
                                        metric=case.get("metric"), horizon=case.get("horizon"), allow_fallback=False)
            pages = [(h["document_id"], h["page"]) for h in result["hits"]]
            row = {"id": case["id"], "elapsed_ms": result["trace"]["elapsed_ms"],
                   "retrieval_run_id": result["retrieval_run_id"], "actual_pages": pages}
            if "expected_pages" in case:
                expected = {tuple(pair) for pair in case["expected_pages"]}
                ranks = [i + 1 for i, pair in enumerate(pages) if pair in expected]
                ideal = sum(1 / math.log2(i + 2) for i in range(min(len(expected), len(pages))))
                row.update({"kind": "page_retrieval", "passed": bool(ranks), "expected_pages": sorted(expected),
                            "recall_at_k": len(set(pages) & expected) / len(expected),
                            "reciprocal_rank": 1 / ranks[0] if ranks else 0,
                            "ndcg_at_k": sum(1 / math.log2(rank + 1) for rank in ranks) / ideal if ideal else 0})
            elif case.get("expected_no_hits"):
                row.update({"kind": "scope_control", "passed": not pages})
            else:
                actual = [c["annotation_id"] for c in result["bank_claims"]]
                row.update({"kind": "claim_control", "passed": set(actual) == set(case["expected_claims"])
                            and result["bank_view_status"] == case["expected_bank_status"],
                            "actual_claims": actual, "bank_view_status": result["bank_view_status"]})
            results.append(row)
        positives = [r for r in results if r["kind"] == "page_retrieval"]
        controls = [r for r in results if r["kind"] != "page_retrieval"]
        reports[mode] = {
            "passed_cases": sum(r["passed"] for r in results), "total_cases": len(results),
            "page_hit_rate_at_5": sum(r["passed"] for r in positives) / len(positives),
            "mean_recall_at_5": float(np.mean([r["recall_at_k"] for r in positives])),
            "mrr_at_5": float(np.mean([r["reciprocal_rank"] for r in positives])),
            "mean_ndcg_at_5": float(np.mean([r["ndcg_at_k"] for r in positives])),
            "controls_passed": sum(r["passed"] for r in controls), "controls_total": len(controls),
            "p50_ms": float(np.percentile([r["elapsed_ms"] for r in results], 50)),
            "p95_ms": float(np.percentile([r["elapsed_ms"] for r in results], 95)), "results": results,
        }
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {
        "scope": "open_development_set_not_holdout", "cases_sha256": file_hash(cases_path),
        "index_version": retriever.manifest["version"], "corpus_version": retriever.manifest["corpus_version"],
        "embedding_space_id": retriever.manifest["embedding_space_id"], "encoder": encoder.space,
        "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine()},
        "timing": "one pass per mode; warm encoder; query cache cleared before each case; model/index load excluded",
        "evaluation_peak_rss_bytes": rss if sys.platform == "darwin" else rss * 1024,
        "index_build": read_json(Path(index_dir) / "metrics.json"), "modes": reports,
    }
