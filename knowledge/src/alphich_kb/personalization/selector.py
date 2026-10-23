"""Faceted retrieval, sourced applicability, bounded ranking and feed diversity."""

import copy
import math
import time
from collections import Counter, defaultdict

from ..build import chunk_spans
from ..common import KnowledgeError, digest
from ..search import LexicalIndex, lexical_candidates
from . import VERSION
from .facets import build_facets
from .inputs import SelectionScope, instant, normalize_event, words
from .planning import bank_questions
from .profiles import normalize_company
from .rules import BUSINESS, match_reasons
from .reprints import group_reprints
from .grading import MIN_SELECTED_GRADE, VERSION as GRADING_VERSION, grade_event


class NewsSelector:
    """Reuse this object and its local embeddings across companies in a worker."""

    def __init__(self, events, encoder=None):
        self.events = []
        identities, source_versions, fact_versions = {}, {}, {}
        for row in events:
            event = normalize_event(row)
            for source in event["sources"]:
                identity = (source["id"], source["version"])
                signature = digest(source)
                if identity in source_versions and source_versions[identity] != signature:
                    raise KnowledgeError("A source ID/version has contradictory immutable content")
                source_versions[identity] = signature
            for fact in event["facts"]:
                signature = digest(fact)
                if fact["id"] in fact_versions and fact_versions[fact["id"]] != signature:
                    raise KnowledgeError("A fact ID is attached to different immutable evidence")
                fact_versions[fact["id"]] = signature
            key = (event["id"], event["version"])
            if key in identities and identities[key] != event["snapshot_hash"]:
                raise KnowledgeError("An event ID/version has contradictory immutable snapshots")
            if key not in identities:
                self.events.append(event)
            identities[key] = event["snapshot_hash"]
        self.encoder, self.vectors = encoder, None
        self.encoder_error = None
        self.embedding_records = []
        if encoder is not None and self.events:
            import numpy as np
            from ..embeddings import EncoderUnavailable
            try:
                cached = encoder.prepared_embeddings(self.events) if hasattr(encoder, "prepared_embeddings") else None
                if cached is not None:
                    self.embedding_records, self.vectors = cached
                else:
                    self._prepare_embeddings(encoder)
                rows = max((r["vector_row"] for r in self.embedding_records), default=-1) + 1
                if self.vectors.shape != (rows, encoder.dimension) or not np.isfinite(self.vectors).all():
                    raise KnowledgeError("Invalid news embedding matrix")
                if not np.allclose(np.linalg.norm(self.vectors, axis=1), 1, atol=1e-5):
                    raise KnowledgeError("News embeddings must be normalized")
            except EncoderUnavailable:
                self.encoder, self.encoder_error = None, "encoder_unavailable"

    def _prepare_embeddings(self, encoder):
        prepared, unique = [], {}
        counter = type("Counter", (), {"count": lambda _, text: encoder.count_tokens(text)})()
        for event in self.events:
            for source in event["sources"]:
                header = f"passage: Новость: {event['title'][:180]}\n"
                for start, end in chunk_spans(source["text"], header, counter, max_tokens=384):
                    text = header + source["text"][start:end]
                    key = digest(text)
                    if key not in unique:
                        unique[key] = len(prepared)
                        prepared.append(text)
                    self.embedding_records.append({"event_id": event["id"], "event_version": event["version"],
                        "snapshot_hash": event["snapshot_hash"], "source_id": source["id"],
                        "source_version": source["version"], "start": start, "end": end,
                        "input_hash": key, "embedding_space_id": encoder.space_id, "vector_row": unique[key]})
        self.vectors = encoder.encode_passages(prepared, batch_size=4)

    def _visible(self, scope):
        excluded, versions = Counter(), defaultdict(list)
        for event in self.events:
            reason = scope.excludes(event)
            if reason:
                excluded[reason] += 1
                continue
            versions[event["id"]].append(event)
        visible = []
        for snapshots in versions.values():
            latest = max(snapshots, key=lambda e: (instant(e["available_at"]), int(e["version"]) if e["version"].isdigit() else -1, e["version"]))
            current = copy.deepcopy(latest)
            current["facts"] = [f for f in current["facts"] if instant(f["source_available_at"]) <= instant(scope.as_of)]
            current["age_anchor"] = min((s["published_at"] or s["available_at"] for s in snapshots), key=instant)
            visible.append(current)
            excluded["older_event_version"] += len(snapshots) - 1
        result, reprints = group_reprints(visible)
        excluded.update(reprints)
        return sorted(result, key=lambda e: e["id"]), dict(excluded)

    def _retrieve(self, events, facets, mode):
        lexical = {}
        with LexicalIndex(events) as index:
            for facet in facets:
                ranked = lexical_candidates(events, facet["text"], index=index)
                lexical[facet["id"]] = {events[row]["snapshot_hash"]: rank for rank, (row, _) in enumerate(ranked[:15], 1)}
        vectors, actual_mode, error = {}, mode, None
        if mode in {"dense", "hybrid"}:
            if self.encoder is None or self.vectors is None:
                actual_mode, error = "lexical_fallback", "encoder_unavailable"
            else:
                from ..embeddings import EncoderUnavailable
                try:
                    vectors = {f["id"]: self.encoder.encode_query(f["text"]) for f in facets}
                except EncoderUnavailable:
                    actual_mode, error = "lexical_fallback", "encoder_unavailable"
        allowed = {e["snapshot_hash"] for e in events}
        records = [r for r in self.embedding_records if r["snapshot_hash"] in allowed]
        matches = defaultdict(list)
        for facet in facets:
            lex = lexical[facet["id"]] if actual_mode in {"lexical", "hybrid", "lexical_fallback"} else {}
            dense, cosine = {}, {}
            if facet["id"] in vectors and records:
                scores = self.vectors[[r["vector_row"] for r in records]] @ vectors[facet["id"]]
                for record, score in zip(records, scores):
                    key = record["snapshot_hash"]
                    cosine[key] = max(cosine.get(key, -1.0), float(score))
                dense = {key: rank for rank, (key, _) in enumerate(sorted(cosine.items(), key=lambda p: (-p[1], p[0]))[:15], 1)}
            denominator = 2 / 61 if actual_mode == "hybrid" else 1 / 61
            for key in set(lex) | set(dense):
                score = ((1 / (60 + lex[key]) if key in lex else 0) + (1 / (60 + dense[key]) if key in dense else 0))
                matches[key].append({"facet_id": facet["id"], "kind": facet["kind"], "feature_ids": facet["feature_ids"],
                                     "lexical_rank": lex.get(key), "dense_rank": dense.get(key), "cosine": cosine.get(key),
                                     "rrf": score, "bounded_rank_score": score / denominator})
        return matches, actual_mode, error

    def select(self, profile, scope, *, limit=5, mode="hybrid", max_per_topic=2, min_grade=0):
        started = time.perf_counter()
        if not isinstance(scope, SelectionScope) or not 1 <= limit <= 10 or not 1 <= max_per_topic <= 10:
            raise KnowledgeError("A server SelectionScope and limits in 1..10 are required")
        if mode not in {"lexical", "dense", "hybrid"}:
            raise KnowledgeError("Mode must be lexical, dense or hybrid")
        if type(min_grade) is not int or not 0 <= min_grade <= 5:
            raise KnowledgeError("min_grade must be an integer in 0..5")
        company = normalize_company(profile)
        if company["id"] != scope.company_id or company["role"] not in {None, scope.role}:
            raise KnowledgeError("Profile does not belong to the job's company/role scope")
        facets = build_facets(company, scope.role)
        events, exclusions = self._visible(scope)
        matches, actual_mode, fallback = self._retrieve(events, facets, mode)
        ranked, review = [], []
        for event in events:
            reasons, topics = match_reasons(company, event, scope)
            retrieval = matches.get(event["snapshot_hash"], [])
            if not reasons:
                if event["facts"] and topics and not topics & BUSINESS:
                    continue
                contextual = [r for r in retrieval if r["kind"] in {
                    "industry", "trade", "finance", "payments", "employment", "eligibility"}]
                if contextual:
                    best = max(contextual, key=lambda r: (r["bounded_rank_score"], r["facet_id"]))
                    review.append({"event_id": event["id"], "event_version": event["version"],
                                   "status": "needs_applicability_check", "relevance_tier": "hypothesis",
                                   "reason_codes": ["retrieval_match_without_structural_evidence"],
                                   "matched_feature_ids": best["feature_ids"], "matched_fact_ids": [],
                                   "semantic_facet_ids": [best["facet_id"]], "ranking": best,
                                   "limitations": ["retrieval_rank_is_not_applicability", "do_not_publish_as_validated_feed"]})
                continue
            strongest = max(reasons, key=lambda r: (r["base"], r["code"]))
            grading = grade_event(company, event, scope, reasons)
            hypothesis = strongest["tier"] == "hypothesis"
            if hypothesis and event["facts"]:
                review.append({"event_id": event["id"], "event_version": event["version"], "status": "needs_applicability_check",
                               **grading,
                               "relevance_tier": "hypothesis", "reason_codes": sorted({r["code"] for r in reasons}),
                               "matched_feature_ids": sorted({i for r in reasons for i in r["feature_ids"]}),
                               "matched_fact_ids": sorted({i for r in reasons for i in r["fact_ids"]}),
                               "semantic_facet_ids": [r["facet_id"] for r in retrieval],
                               "ranking": max(retrieval, key=lambda r: r["bounded_rank_score"]) if retrieval else {"bounded_rank_score": 0},
                               "limitations": sorted({x for r in reasons for x in r["limitations"]})})
                continue
            age = max(0, (instant(scope.as_of) - instant(event["age_anchor"])).total_seconds() / 86400)
            half_life = 90 if "regulation" in topics else 30 if "policy_rate" in topics else 14
            freshness = 5 * math.pow(0.5, age / half_life)
            retrieval_bonus = 4 * max((r["bounded_rank_score"] for r in retrieval), default=0)
            finance = any(r["angle"] in {"financing_cost", "cash_placement_yield", "lease_financing"} for r in reasons)
            role_bonus = 4 if (scope.role == "cfo" and finance) or (scope.role == "ceo" and any(r["code"] == "industry_scope_match" for r in reasons)) else 0
            priority = min(100, strongest["base"] + freshness + retrieval_bonus + role_bonus)
            fields = {"event_id": event["id"], "event_version": event["version"], "snapshot_hash": event["snapshot_hash"],
                      **grading,
                      "title": event["title"], "published_at": event["published_at"], "available_at": event["available_at"],
                      "priority": round(priority, 3), "relevance_tier": strongest["tier"],
                      "status": "needs_extraction" if hypothesis and not event["facts"] else "needs_applicability_check" if hypothesis else "applicable_with_conditions",
                      "reason_codes": sorted({r["code"] for r in reasons}), "reasons": reasons,
                      "matched_feature_ids": sorted({i for r in reasons for i in r["feature_ids"]}),
                      "matched_fact_ids": sorted({i for r in reasons for i in r["fact_ids"]}),
                      "semantic_facet_ids": sorted({m["facet_id"] for m in retrieval}),
                      "coverage": [{"feature_id": r["feature_ids"][0], "known_profile_amount_share": r["coverage"],
                                    "share_of_all_flow": r["share_of_all_flow"], "direction": r["direction"]}
                                   for r in reasons if r["code"] == "payment_neighbor_sector"],
                      "limitations": sorted(set(company["limitations"] + ["relevance_is_not_financial_impact"] + [x for r in reasons for x in r["limitations"]]
                                                + (["synthetic_publication"] if any(s["synthetic"] for s in event["sources"]) else [])
                                                + (["publication_date_unknown"] if not event["published_at"] else []))),
                      "bank_questions": bank_questions(company, event, reasons, topics, scope),
                      "primary_topic": "policy_rate" if "policy_rate" in topics else next(iter(sorted(topics)), "industry"),
                      "major_event": any(f["attributes"].get("kind") == "rate_decision" and f["attributes"].get("geography") == scope.market_country for f in event["facts"]),
                      "duplicate_event_ids": event["duplicate_event_ids"],
                      "story_group_id": event["story_group_id"], "reprint_sources": event["reprint_sources"],
                      "reprint_group_size": event["reprint_group_size"],
                      "ranking": {"structural_base": strongest["base"], "freshness_bonus": freshness,
                                  "retrieval_bonus": retrieval_bonus, "role_bonus": role_bonus, "facets": retrieval}}
            ranked.append(fields)
        # Candidate retrieval preserves its historical order for manual analysis;
        # a curated feed orders by grade first and never fills gaps with <=3.
        ranked.sort(key=lambda item: (-item["relevance_grade"] if min_grade else 0, -item["priority"], item["event_id"]))
        selected, skipped, topic_counts = [], [], Counter()
        for item in ranked:
            if item["relevance_grade"] < min_grade:
                continue
            if len(selected) >= limit:
                break
            if (topic_counts[item["primary_topic"]] >= max_per_topic and not item["major_event"]
                    and item["relevance_grade"] < 5):
                skipped.append(item["event_id"])
                continue
            topic_counts[item["primary_topic"]] += 1
            selected.append(item)
        trace = {"company_id": company["id"], "profile_version": company["profile_version"], "context_version": company["context_version"],
                 "profile_input_hash": company["input_hash"], "role": scope.role, "as_of": scope.as_of,
                 "market_country_policy": scope.market_country, "method_version": VERSION, "requested_mode": mode,
                 "mode": actual_mode, "fallback_reason": fallback, "limit": limit, "max_per_topic": max_per_topic,
                 "min_grade": min_grade, "selection_threshold": MIN_SELECTED_GRADE, "grading_version": GRADING_VERSION,
                 "below_grade_threshold": sum(r["relevance_grade"] < min_grade for r in ranked),
                 "embedding_space_id": self.encoder.space_id if self.encoder else None,
                 "visible_snapshots": sorted(e["snapshot_hash"] for e in events), "exclusion_counts": exclusions,
                 "rejected_fact_count": sum(len(e["rejected_facts"]) for e in events),
                 "diversity_skipped_event_ids": skipped, "selected_event_ids": [r["event_id"] for r in selected],
                 "extraction_candidate_ids": [r["event_id"] for r in ranked if r["status"] == "needs_extraction"][:3]}
        run_id = f"news_selection_{digest(trace)}"
        for item in selected:
            item["retrieval_run_id"] = run_id
        trace["elapsed_ms"] = round((time.perf_counter() - started) * 1000, 3)
        review.sort(key=lambda item: (-item["ranking"]["bounded_rank_score"], item["event_id"]))
        return {"company_id": company["id"], "role": scope.role, "items": selected, "review_queue": review[:3],
                "grade_scale": {"min": 0, "max": 5}, "selection_threshold": MIN_SELECTED_GRADE,
                "min_grade": min_grade,
                "extraction_candidates": [r for r in ranked if r["status"] == "needs_extraction"][:3],
                "facets": facets, "mode": actual_mode, "retrieval_run_id": run_id, "trace": trace}
