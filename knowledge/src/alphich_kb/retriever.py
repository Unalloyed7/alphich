"""Hybrid source retrieval with server-owned scope and bounded evidence context."""

import copy
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np

from .common import KnowledgeError, digest, read_jsonl, verify_artifacts
from .embeddings import EncoderUnavailable
from .search import available, lexical_candidates, select_claims

RANKING_VERSION = "page-rrf-v1"


@dataclass(frozen=True)
class RetrievalScope:
    """Constructed by the server from the job and permissions, never by an LLM."""
    as_of: str
    allowed_document_ids: frozenset[str]

    def __post_init__(self):
        date.fromisoformat(self.as_of)
        if isinstance(self.allowed_document_ids, str):
            raise KnowledgeError("Document access scope must be a collection of IDs")
        if any(not isinstance(value, str) for value in self.allowed_document_ids):
            raise KnowledgeError("Document access IDs must be strings")
        object.__setattr__(self, "allowed_document_ids", frozenset(self.allowed_document_ids))

    def permits(self, item):
        return item["document_id"] in self.allowed_document_ids and available(item, self.as_of)


def collapse_pages(ranked, items, limit):
    result, seen = [], set()
    for row, score in ranked:
        parent = items[row]["parent_id"]
        if parent in seen:
            continue
        seen.add(parent)
        result.append({"row": row, "parent_id": parent, "rank": len(result) + 1, "score": float(score)})
        if len(result) >= limit:
            break
    return result


def reciprocal_rank_fusion(lists, k=60):
    """One contribution per page per modality, independent of raw score scales."""
    combined = {}
    for name, ranked in lists.items():
        for item in ranked:
            record = combined.setdefault(item["parent_id"], {"parent_id": item["parent_id"], "rrf": 0.0, "matches": {}})
            record["rrf"] += 1 / (k + item["rank"])
            record["matches"][name] = item
    return sorted(combined.values(), key=lambda r: (-r["rrf"], r["parent_id"]))


def horizon_end(horizon):
    if not horizon:
        return None
    if "-Q" in horizon:
        year, quarter = horizon.split("-Q")
        month = int(quarter) * 3
        return date(int(year), month, 31 if month in {3, 12} else 30)
    try:
        return date.fromisoformat(horizon.split("/")[-1])
    except ValueError:
        return None


class BankRetriever:
    """Load one index and share one encoder across requests in a worker process."""

    def __init__(self, index_dir, encoder=None):
        root = Path(index_dir)
        self.manifest = verify_artifacts(root)
        if self.manifest["kind"] != "bank_retrieval_index":
            raise KnowledgeError("Expected a bank retrieval index")
        self.items = read_jsonl(root / "items.jsonl")
        self.documents = {d["id"]: d for d in read_jsonl(root / "documents.jsonl")}
        self.pages = {p["id"]: p for p in read_jsonl(root / "pages.jsonl")}
        self.claims = read_jsonl(root / "claims.jsonl")
        self.by_id = {item["id"]: item for item in self.items}
        self.vectors = np.load(root / "vectors.npy", allow_pickle=False, mmap_mode="r")
        expected_shape = (len(self.items), self.manifest["dimension"])
        if len(self.by_id) != len(self.items) or self.vectors.shape != expected_shape:
            raise KnowledgeError("Index matrix and evidence rows do not agree")
        if self.vectors.dtype != np.float32 or not np.isfinite(self.vectors).all():
            raise KnowledgeError("Index matrix must contain finite float32 vectors")
        if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
            raise KnowledgeError("Index matrix is not normalized")
        for item in self.items:
            doc = self.documents.get(item["document_id"])
            parent = self.pages.get(item["parent_id"])
            if not doc or not parent or parent["document_id"] != item["document_id"]:
                raise KnowledgeError("Broken document/parent link in retrieval index")
            span = item["source_span"]
            if item["published_at"] != doc["published_at"] or parent["text"][span["start"]:span["end"]] != item["text"]:
                raise KnowledgeError("Index evidence does not agree with its source")
        for claim in self.claims:
            evidence = self.by_id.get(claim["document_chunk_id"])
            if evidence is None or evidence["document_id"] != claim["document_id"] or not evidence["ocr_checked"]:
                raise KnowledgeError("Verified claim has no corresponding reviewed source evidence")
        self.encoder = encoder
        if encoder is not None and encoder.space_id != self.manifest["embedding_space_id"]:
            raise KnowledgeError("Query encoder does not match the index embedding space")
        self.expanded = OrderedDict()

    def get_evidence(self, evidence_id, scope):
        item = self.by_id.get(evidence_id) or self.expanded.get(evidence_id)
        if item is None or not scope.permits(item):
            raise KnowledgeError("Evidence is unavailable in this retrieval scope")
        return copy.deepcopy(item)

    def _expand(self, item, scope, maximum=4000):
        if not scope.permits(item):
            raise KnowledgeError("Cannot expand evidence outside the retrieval scope")
        page = self.pages[item["parent_id"]]
        if page["document_id"] != item["document_id"]:
            raise KnowledgeError("Parent belongs to a different document")
        span = item["source_span"]
        text = page["text"]
        if text[span["start"]:span["end"]] != item["text"]:
            raise KnowledgeError("Evidence span no longer matches its parent")
        start, end = max(0, span["start"] - 600), min(len(text), span["end"] + 600)
        if end - start > maximum:
            start, end = span["start"], span["end"]
        if end - start > maximum:
            return None
        identity = digest({"page_text_version": page["text_version"], "start": start, "end": end, "kind": "parent_window_v1"})
        context = {
            "id": f"span_{identity}", "kind": "document_chunk", "document_id": item["document_id"],
            "source_version": item["source_version"], "parent_id": page["id"], "page": page["page"],
            "published_at": item["published_at"], "url": item["url"], "title": item["title"],
            "content_hash": item["content_hash"], "text": text[start:end], "text_version": identity,
            "source_span": {"start": start, "end": end, "unit": "unicode_codepoints"},
            "ocr_checked": False, "extraction_status": "ocr_unreviewed",
            "quality_flags": page["quality_flags"], "allowed_uses": ["locate_source", "unverified_context"],
            "source_refs": [{**item["source_refs"][0], "start": start, "end": end}],
        }
        self.expanded[context["id"]] = context
        while len(self.expanded) > 256:
            self.expanded.popitem(last=False)
        return context

    def _claims(self, scope, metric, horizon):
        permitted = [c for c in self.claims if scope.permits(c)]
        if not metric:
            return [], []
        matching = [c for c in permitted if c["metric"] == metric and c["attributed_to"] == "alfa_bank"
                    and c["geography"] == "RU" and c["scenario"] == "base"]
        if horizon:
            chosen = select_claims(matching, scope.as_of, metric, horizon)
        else:
            chosen = [c for c in matching if c["kind"] != "forecast" and c["position_role"] == "current_in_document"
                      and c["verification_status"] == "visually_verified"]
        selected_ids = {c["id"] for c in chosen}
        rejected = []
        for claim in matching:
            if claim["id"] in selected_ids:
                continue
            reason = "superseded_or_unverified"
            if claim["position_role"] != "current_in_document":
                reason = "previous_or_conditional_view"
            elif claim["kind"] == "forecast" and claim["forecast_horizon"] != horizon:
                reason = "wrong_or_unspecified_horizon"
            rejected.append({"claim_id": claim["id"], "reason": reason})
        return chosen, rejected

    def retrieve(self, query, scope, *, mode="dense", limit=5, topics=None, metric=None, horizon=None,
                 max_context_chars=16000, allow_fallback=True):
        started = time.perf_counter()
        if not isinstance(query, str) or not query.strip() or len(query) > 8000:
            raise KnowledgeError("Query must contain 1..8000 characters")
        if mode not in {"lexical", "dense", "hybrid"} or not 1 <= limit <= 10:
            raise KnowledgeError("Mode must be lexical/dense/hybrid and limit must be 1..10")
        if not 512 <= max_context_chars <= 40000 or (horizon and not metric):
            raise KnowledgeError("Context budget must be 512..40000; horizon requires a metric")
        if not isinstance(scope, RetrievalScope):
            raise KnowledgeError("A server-owned RetrievalScope is required")
        eligible_rows = [i for i, item in enumerate(self.items) if scope.permits(item)
                         and (not topics or set(topics).intersection(item["topics"]))]
        eligible_items = [self.items[i] for i in eligible_rows]
        pool_size = max(10, limit * 2)
        lexical = []
        if mode in {"lexical", "hybrid"}:
            lexical = collapse_pages([(eligible_rows[row], score) for row, score in lexical_candidates(eligible_items, query)],
                                     self.items, pool_size)
        dense, actual_mode, fallback_reason = [], mode, None
        if mode in {"dense", "hybrid"} and eligible_rows:
            try:
                if self.encoder is None:
                    raise EncoderUnavailable("Encoder was not loaded")
                query_vector = self.encoder.encode_query(query)
                scores = self.vectors[eligible_rows] @ query_vector
                if not np.isfinite(scores).all():
                    raise KnowledgeError("Dense search returned non-finite scores")
                ranked = sorted(zip(eligible_rows, scores), key=lambda pair: (-float(pair[1]), self.items[pair[0]]["id"]))
                dense = collapse_pages(ranked, self.items, pool_size)
            except EncoderUnavailable:
                if not allow_fallback:
                    raise
                actual_mode, fallback_reason = "lexical_fallback", "encoder_unavailable"
                lexical = collapse_pages([(eligible_rows[row], score) for row, score in lexical_candidates(eligible_items, query)],
                                         self.items, pool_size)
        elif mode in {"dense", "hybrid"} and self.encoder is None:
            actual_mode, fallback_reason = "lexical_fallback", "encoder_unavailable"
        ranked_pages = reciprocal_rank_fusion({"lexical": lexical, "dense": dense})[:limit]
        hits = []
        for page in ranked_pages:
            matches = page["matches"]
            representative = min(matches.items(), key=lambda pair: (pair[1]["rank"], pair[0]))[1]
            item = self.items[representative["row"]]
            age = (date.fromisoformat(scope.as_of) - date.fromisoformat(item["published_at"])).days
            hits.append({"evidence_id": item["id"], "document_id": item["document_id"], "parent_id": item["parent_id"],
                         "page": item["page"], "title": item["title"], "url": item["url"], "published_at": item["published_at"],
                         "text": item["text"], "ocr_checked": item["ocr_checked"], "quality_flags": item["quality_flags"],
                         "source_age_days": age, "rrf": page["rrf"],
                         "matches": {name: {"rank": match["rank"], "score": match["score"],
                                             "evidence_id": self.items[match["row"]]["id"]} for name, match in matches.items()}})
        chosen_claims, rejected_claims = self._claims(scope, metric, horizon)
        evidence, contexts, included, used_chars = [], [], set(), 0
        omitted = []

        def include(item, destination):
            nonlocal used_chars
            if item is None or item["id"] in included:
                return
            if used_chars + len(item["text"]) > max_context_chars:
                omitted.append(item["id"])
                return
            included.add(item["id"])
            used_chars += len(item["text"])
            destination.append(copy.deepcopy(item))

        # Registry evidence takes precedence, including revisions missed by top-K.
        for claim in chosen_claims:
            include(self.get_evidence(claim["document_chunk_id"], scope), evidence)
        for hit in hits:
            for match in hit["matches"].values():
                include(self.get_evidence(match["evidence_id"], scope), evidence)
        for hit in hits:
            if hit["evidence_id"] in included:
                include(self._expand(self.by_id[hit["evidence_id"]], scope), contexts)
        chosen_claims = [c for c in chosen_claims if c["document_chunk_id"] in included]
        end = horizon_end(horizon)
        historical = bool(end and end < date.fromisoformat(scope.as_of))
        bank_status = "historical_only" if chosen_claims and historical else "found" if chosen_claims else "not_found"
        if horizon and len({(c["value"], c["unit"]) for c in chosen_claims}) > 1:
            bank_status = "conflicting"
        limitations = ["semantic_applicability_requires_analysis", "publication_time_precision_is_day",
                       "unreviewed_ocr_cannot_confirm_exact_numbers"]
        if not metric:
            limitations.append("claim_selection_requires_metric_and_forecast_horizon")
        if omitted:
            limitations.append("context_budget_exhausted")
        if historical:
            limitations.append("forecast_period_elapsed")
        trace = {"query": query, "as_of": scope.as_of, "allowed_document_ids": sorted(scope.allowed_document_ids),
                 "topics": sorted(topics or []), "metric": metric, "horizon": horizon,
                 "requested_mode": mode, "mode": actual_mode, "fallback_reason": fallback_reason,
                 "index_version": self.manifest["version"], "corpus_version": self.manifest["corpus_version"],
                 "embedding_space_id": self.manifest["embedding_space_id"], "ranking_version": RANKING_VERSION,
                 "rrf_k": 60, "eligible_items": len(eligible_rows), "limit": limit,
                 "max_context_chars": max_context_chars, "used_context_chars": used_chars,
                 "omitted_evidence_ids": omitted, "rejected_claims": rejected_claims,
                 "selected_evidence_ids": [e["id"] for e in evidence + contexts],
                 "selected_claim_ids": [c["id"] for c in chosen_claims]}
        run_id = f"retrieval_{digest(trace)}"
        trace["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        return {"retrieval_run_id": run_id, "status": "evidence_found" if chosen_claims else "candidates_only" if hits else "not_found",
                "mode": actual_mode, "bank_view_status": bank_status, "hits": hits,
                "bank_claims": copy.deepcopy(chosen_claims), "evidence": evidence, "parent_contexts": contexts,
                "limitations": limitations, "trace": trace}
