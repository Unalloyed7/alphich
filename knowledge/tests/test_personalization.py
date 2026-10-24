import copy
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from alphich_kb.common import KnowledgeError
from alphich_kb.personalization.facets import build_facets
from alphich_kb.personalization.inputs import SelectionScope, from_backend_event, normalize_event
from alphich_kb.personalization.profiles import normalize_company
from alphich_kb.personalization.selector import NewsSelector

try:
    import numpy as np
except ImportError:
    np = None

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PROFILES = json.loads((FIXTURES / "company_profiles.json").read_text())["companies"]
EVENTS = json.loads((FIXTURES / "news_events.json").read_text())["events"]
AS_OF = "2026-10-20T12:00:00Z"


def profile(uid="S_BUILD"):
    return copy.deepcopy(next(p for p in PROFILES if p.get("company_id", p.get("id")) == uid))


def event(uid="rate_decision", version=1):
    return copy.deepcopy(next(e for e in EVENTS if e["id"] == uid and e["version"] == version))


def scope(uid="S_BUILD", allowed=None, when=AS_OF, role="cfo"):
    return SelectionScope(uid, when, frozenset(e["id"] for e in EVENTS) if allowed is None else frozenset(allowed), role)


def select(rows, uid="S_BUILD", company=None, selection_scope=None, **kwargs):
    return NewsSelector(rows).select(company or profile(uid), selection_scope or scope(uid), mode="lexical", **kwargs)


def renamed(row, uid):
    row = copy.deepcopy(row)
    row["id"] = uid
    for source in row["sources"]:
        original = source["id"]
        source["id"] = uid + ":" + original
        for fact in row["facts"]:
            if fact["source_id"] == original:
                fact["source_id"] = source["id"]
                fact["id"] = uid + ":" + fact["id"]
    return row


class PersonalizationTests(unittest.TestCase):
    def test_open_business_effects_survive_source_bound_fact_normalization(self):
        for effect in ("reputation", "cybersecurity", "safety", "accessibility", "customer_trust"):
            with self.subTest(effect=effect):
                row = event()
                row["facts"][0].setdefault("attributes", {})["business_effect"] = effect
                normalized = normalize_event(row)
                self.assertEqual(normalized["facts"][0]["attributes"]["business_effect"], effect)
        for effect in ("", "x" * 65, "invalid effect", ["reputation"]):
            with self.subTest(invalid_effect=effect):
                row = event()
                row["facts"][0].setdefault("attributes", {})["business_effect"] = effect
                normalized = normalize_event(row)
                self.assertNotIn(row["facts"][0]["id"], {f["id"] for f in normalized["facts"]})
                self.assertIn("invalid_business_effect", {f["reason"] for f in normalized["rejected_facts"]})

    def test_facets_share_one_eligible_index_without_changing_ranking(self):
        from alphich_kb.personalization import selector as module
        from alphich_kb.search import LexicalIndex, lexical_candidates
        rows = [event(), event("construction")]
        with patch.object(module, "LexicalIndex", wraps=LexicalIndex) as index:
            optimized = select(rows)
        self.assertEqual(index.call_count, 1)
        self.assertGreater(len(optimized["facets"]), 1)
        # Reference behavior: a fresh FTS index for every individual query.
        def reference(selected, query, **_):
            return lexical_candidates(selected, query)
        with patch.object(module, "lexical_candidates", side_effect=reference):
            expected = select(rows)
        for result in (optimized, expected):
            result["trace"].pop("elapsed_ms")
        self.assertEqual(optimized, expected)

    def test_rate_has_different_credit_and_deposit_angles(self):
        credit = select([event()])["items"][0]
        deposit = select([event()], "S_RENT")["items"][0]
        software = select([event()], "S_SOFTWARE")["items"][0]
        self.assertIn("credit_rate_channel", credit["reason_codes"])
        self.assertIn("deposit_rate_channel", deposit["reason_codes"])
        self.assertEqual(software["reason_codes"], ["national_rate_context"])
        self.assertGreater(credit["priority"], software["priority"])
        self.assertNotEqual(credit["bank_questions"][1]["query"], deposit["bank_questions"][1]["query"])
        self.assertIn("credit_balance_unknown", credit["limitations"])
        for item, uid in [(credit, "S_BUILD"), (deposit, "S_RENT")]:
            self.assertTrue(set(item["matched_feature_ids"]) <= {f["id"] for f in profile(uid)["features"]})
            self.assertTrue(item["matched_fact_ids"])

    def test_facets_are_bounded_and_false_products_are_not_queries(self):
        for uid in ("S_BUILD", "S_RENT", "S_SOFTWARE", "S_FAB", "S_GRAIN"):
            facets = build_facets(normalize_company(profile(uid)), "cfo")
            self.assertLessEqual(len(facets), 7)
        software = build_facets(normalize_company(profile("S_SOFTWARE")), "cfo")
        self.assertNotIn("finance", [f["kind"] for f in software])
        self.assertNotIn("trade", [f["kind"] for f in software])

    def test_both_company_contracts_keep_existing_feature_ids(self):
        for uid in ("S_BUILD", "S_GRAIN"):
            raw = profile(uid)
            normalized = normalize_company(raw)
            used = {i for ids in normalized["bindings"].values() for i in ids}
            self.assertTrue(used)
            self.assertTrue(used <= {f["id"] for f in raw["features"]})

    def test_wrong_feature_version_owner_or_value_is_rejected(self):
        for mutation in ("version", "owner", "value"):
            raw = profile("S_GRAIN")
            feature = next(f for f in raw["features"] if f["name"] == "CL_PRODUCT_LOA_MARK")
            if mutation == "version":
                feature["profile_version"] = "stale"
            elif mutation == "owner":
                feature["company_id"] = "other"
            else:
                feature["value"] = False
            with self.subTest(mutation=mutation), self.assertRaises(KnowledgeError):
                normalize_company(raw)

    def test_product_without_feature_evidence_does_not_create_direct_reason(self):
        raw = profile()
        raw["features"] = [f for f in raw["features"] if f["name"] != "products"]
        item = select([event()], company=raw)["items"][0]
        self.assertEqual(item["reason_codes"], ["national_rate_context"])

    def test_foreign_trade_requires_business_topic_and_sector_scope(self):
        rows = [event(i) for i in ("grain_trade", "china_customs", "china_politics", "china_sports")]
        build = select(rows)
        grain = select(rows, "S_GRAIN")
        self.assertEqual([i["event_id"] for i in build["items"]], ["china_customs"])
        self.assertEqual({i["event_id"] for i in grain["items"]}, {"grain_trade", "china_customs"})

    def test_fts_false_with_country_list_does_not_prove_trade(self):
        raw = profile("S_GRAIN")
        raw["fts_flag"] = False
        result = select([event("china_customs")], "S_GRAIN", company=raw)
        self.assertEqual(result["items"], [])
        self.assertIn("fts_country_without_flag", normalize_company(raw)["limitations"])

    def test_industry_match_and_foreign_geography_are_distinct(self):
        rows = [event("construction"), event("foreign_build")]
        result = select(rows)
        self.assertEqual([i["event_id"] for i in result["items"]], ["construction"])
        foreign = next(i for i in result["review_queue"] if i["event_id"] == "foreign_build")
        self.assertIn("foreign_operational_presence_unknown", foreign["limitations"])
        self.assertEqual(select([event("construction")], "S_SOFTWARE")["items"], [])

    def test_gold_does_not_imply_exposure_of_metal_structures(self):
        self.assertEqual(select([event("gold")], "S_FAB")["items"], [])
        self.assertEqual(select([event("fed")])["items"], [])

    def test_registration_region_is_only_conditional_context(self):
        row = event("regional_build")
        row["facts"][0]["attributes"]["industry_codes"] = []
        item = select([row])["items"][0]
        self.assertEqual(item["relevance_tier"], "context")
        self.assertIn("registration_is_not_operational_geography", item["limitations"])
        self.assertEqual(select([event("moscow")])["items"], [])

    def test_payment_shares_use_full_flow_denominator_once(self):
        item = select([event("steel_prices")])["items"][0]
        incoming = next(r for r in item["reasons"] if r.get("direction") == "incoming")
        self.assertAlmostEqual(incoming["share_of_all_flow"], 0.6)
        self.assertAlmostEqual(incoming["coverage"], 0.6)
        self.assertAlmostEqual(incoming["base"], 40 + 8 * 0.6)
        self.assertIn("payment_link_business_meaning_unknown", item["limitations"])

    def test_missing_payment_coverage_remains_null_and_wrong_method_is_ignored(self):
        raw = profile()
        for f in raw["features"]:
            if f["name"].startswith("payment_"):
                f["value"]["known_profile_amount_share"] = None
        item = select([event("steel_prices")], company=raw)["items"][0]
        self.assertTrue(all(r["known_profile_amount_share"] is None for r in item["coverage"]))
        for f in raw["features"]:
            if f["name"].startswith("payment_"):
                f["method"] = "untrusted_top_n_denominator"
        self.assertEqual(select([event("steel_prices")], company=raw)["items"], [])

    def test_raw_news_is_extraction_candidate_without_invented_facts(self):
        result = select([event("raw_rate")])
        item = result["items"][0]
        self.assertEqual(item["status"], "needs_extraction")
        self.assertEqual(item["matched_fact_ids"], [])
        self.assertEqual(item["bank_questions"][0]["fact_ids"], [])
        self.assertTrue(any(r["source_refs"] for r in item["reasons"]))

    def test_bad_quote_source_version_or_status_quarantines_annotation(self):
        for mutation in ("quote", "source_version", "status"):
            row = event()
            fact = row["facts"][0]
            if mutation == "quote":
                fact["text"] += " fabricated"
            elif mutation == "source_version":
                fact["source_version"] = "wrong"
            else:
                fact["verification_status"] = "llm_candidate"
            with self.subTest(mutation=mutation):
                result = select([row])
                self.assertEqual(result["items"][0]["status"], "needs_extraction")
                self.assertEqual(result["items"][0]["matched_fact_ids"], [])
                self.assertEqual(result["trace"]["rejected_fact_count"], 1)

    def test_private_and_disallowed_news_are_filtered_before_lexical_search(self):
        private = renamed(event(), "private")
        private["company_id"] = "other"
        private["title"] = "SECRET_PRIVATE_HEADLINE"
        rows = [event(), private, event("construction")]
        from alphich_kb.personalization import selector as module
        original = module.lexical_candidates
        with patch.object(module, "lexical_candidates", wraps=original) as ranker:
            result = select(rows, selection_scope=scope(allowed={"rate_decision", "private"}))
        self.assertTrue(all([e["id"] for e in call.args[0]] == ["rate_decision"] for call in ranker.call_args_list))
        self.assertNotIn("SECRET_PRIVATE_HEADLINE", json.dumps(result))
        self.assertEqual(result["trace"]["exclusion_counts"]["access_denied"], 2)

    def test_empty_access_and_wrong_company_or_role_fail_closed(self):
        self.assertEqual(select([event()], selection_scope=scope(allowed=set()))["items"], [])
        for bad_scope in (scope(uid="other"), scope(role="ceo")):
            raw = profile()
            raw["profile"]["role"] = "cfo"
            with self.assertRaises(KnowledgeError):
                select([event()], company=raw, selection_scope=bad_scope)

    def test_scope_requires_timezone_and_ids_not_a_string(self):
        for when in ("2026-10-20", "2026-10-20T12:00:00", "invalid"):
            with self.assertRaises(KnowledgeError):
                scope(when=when)
        with self.assertRaises(KnowledgeError):
            SelectionScope("S_BUILD", AS_OF, "rate_decision")

    def test_as_of_excludes_unknown_and_future_versions_but_allows_future_law(self):
        rows = [e for e in EVENTS if e["id"] in {"software_update", "future_law", "future_publication", "unknown_availability"}]
        result = select(rows, "S_SOFTWARE")
        self.assertEqual({i["event_id"] for i in result["items"]}, {"software_update", "future_law"})
        self.assertEqual(next(i for i in result["items"] if i["event_id"] == "software_update")["event_version"], "1")
        self.assertEqual(result["trace"]["exclusion_counts"]["availability_unknown"], 1)

    def test_late_received_source_cannot_backdate_an_event_or_fact(self):
        row = event()
        row["sources"][0]["available_at"] = "2026-10-21T10:00:00Z"
        self.assertEqual(select([row])["items"], [])
        row = event()
        row["facts"][0]["source_available_at"] = "2026-10-21T10:00:00Z"
        self.assertEqual(select([row])["items"][0]["status"], "needs_extraction")

    def test_future_source_publication_is_filtered_even_without_event_date(self):
        row = event()
        row["published_at"] = None
        row["sources"][0]["published_at"] = "2026-10-21T00:00:00Z"
        result = select([row])
        self.assertEqual(result["items"], [])
        self.assertEqual(result["trace"]["exclusion_counts"]["future_publication"], 1)

    def test_immutable_event_source_and_fact_id_conflicts_are_rejected(self):
        for mutation in ("event", "source", "fact"):
            first, second = event(), event()
            if mutation == "event":
                second["title"] += " changed"
            elif mutation == "source":
                second["id"] = "other"
                second["sources"][0]["text"] += " changed"
            else:
                second = renamed(second, "other")
                second["facts"][0]["id"] = first["facts"][0]["id"]
            with self.subTest(mutation=mutation), self.assertRaises(KnowledgeError):
                NewsSelector([first, second])

    def test_exact_reprint_uses_one_slot_but_distinct_meeting_does_not(self):
        result = select([event(), event("zz_rate_copy")])
        self.assertEqual(len(result["items"]), 1)
        self.assertEqual(result["items"][0]["duplicate_event_ids"], ["zz_rate_copy"])
        other = renamed(event(), "previous_meeting")
        other["published_at"] = "2026-09-19T10:00:00Z"
        other["sources"][0]["available_at"] = "2026-09-19T10:01:00Z"
        other["facts"][0]["source_available_at"] = other["sources"][0]["available_at"]
        self.assertEqual(len(select([event(), other], selection_scope=scope(allowed={"rate_decision", "previous_meeting"}))["items"]), 2)

    def test_edit_does_not_refresh_original_age(self):
        old = event()
        old["published_at"] = None
        old["sources"][0]["available_at"] = "2026-08-01T00:00:00Z"
        old["facts"][0]["source_available_at"] = "2026-08-01T00:00:00Z"
        updated = renamed(old, "temporary")
        updated["id"], updated["version"] = old["id"], 2
        updated["sources"][0]["available_at"] = "2026-10-19T00:00:00Z"
        updated["facts"][0]["source_available_at"] = "2026-10-19T00:00:00Z"
        item = select([old, updated])["items"][0]
        self.assertEqual(item["event_version"], "2")
        self.assertLess(item["ranking"]["freshness_bonus"], 1)
        self.assertIn("publication_date_unknown", item["limitations"])

    def test_diversity_and_extraction_queue_do_not_fill_with_unproven_items(self):
        rows = [event("rate_forecast"), event("raw_rate"), event("prompt_injection")]
        for n in range(3):
            row = renamed(event("rate_forecast"), "forecast_" + str(n))
            row["title"] += str(n)
            rows.append(row)
        result = select(rows, selection_scope=scope(allowed={r["id"] for r in rows}))
        self.assertEqual(len(result["items"]), 2)
        self.assertTrue(result["trace"]["diversity_skipped_event_ids"])
        self.assertIn("raw_rate", [i["event_id"] for i in result["extraction_candidates"]])
        self.assertNotIn("prompt_injection", [i["event_id"] for i in result["items"]])

    def test_bank_questions_are_bounded_dated_and_horizon_specific(self):
        current = select([event()])["items"][0]
        future = select([event("rate_forecast")])["items"][0]
        self.assertEqual(current["bank_questions"][0]["horizon"], "2026-12-31")
        self.assertEqual(future["bank_questions"][0]["horizon"], "2027-12-31")
        self.assertEqual(current["bank_questions"][0]["as_of"], "2026-10-19")
        self.assertLessEqual(len(current["bank_questions"]), 2)

    def test_backend_adapter_requires_authorized_source_body(self):
        original = event("raw_rate")
        source = {**original["sources"][0], "company_id": "S_BUILD", "received_at": original["sources"][0]["available_at"]}
        metadata = {**original, "company_id": "S_BUILD", "source_ids": [source["id"]]}
        normalized = normalize_event(from_backend_event(metadata, [source]))
        self.assertEqual(normalized["facts"], [])
        for bodies in ([], [{**source, "company_id": "other"}]):
            with self.assertRaises(KnowledgeError):
                from_backend_event(metadata, bodies)


class FakeEncoder:
    dimension = 3
    space_id = "fake_news_space"

    def __init__(self):
        self.passage_calls = 0
        self.inputs = []

    def count_tokens(self, text):
        return len(text.split())

    def encode_passages(self, texts, batch_size=4):
        self.passage_calls += 1
        self.inputs.extend(texts)
        return np.tile(np.array([1, 0, 0], dtype=np.float32), (len(texts), 1))

    def encode_query(self, text):
        return np.array([1, 0, 0], dtype=np.float32)


@unittest.skipIf(np is None, "Install knowledge/requirements-rag.lock for dense tests")
class DensePersonalizationTests(unittest.TestCase):
    def test_embeddings_are_reused_across_companies_with_one_passage_prefix(self):
        encoder = FakeEncoder()
        row = event()
        row["title"] = "passage: query: untrusted title"
        selector = NewsSelector([row], encoder)
        for uid in ("S_BUILD", "S_RENT"):
            selector.select(profile(uid), scope(uid))
        self.assertEqual(encoder.passage_calls, 1)
        self.assertTrue(all(text.startswith("passage: Новость: ") for text in encoder.inputs))

    def test_dense_permission_filter_precedes_top_k_and_no_rank_implies_applicability(self):
        rows = [event("prompt_injection"), event()]
        selector = NewsSelector(rows, FakeEncoder())
        result = selector.select(profile(), scope(allowed={"prompt_injection"}), mode="dense")
        self.assertEqual(result["items"], [])
        self.assertEqual(result["review_queue"][0]["status"], "needs_applicability_check")
        self.assertEqual(result["review_queue"][0]["matched_fact_ids"], [])
        self.assertEqual(result["review_queue"][0]["ranking"]["dense_rank"], 1)

    def test_bonus_is_bounded_by_best_facet_not_sum_of_matches(self):
        selector = NewsSelector([event()], FakeEncoder())
        item = selector.select(profile(), scope())["items"][0]
        self.assertGreater(len(item["ranking"]["facets"]), 1)
        self.assertLessEqual(item["ranking"]["retrieval_bonus"], 4)

    def test_encoder_outage_has_explicit_fallback_but_integrity_error_does_not(self):
        from alphich_kb.embeddings import EncoderUnavailable
        encoder = FakeEncoder()
        selector = NewsSelector([event()], encoder)
        with patch.object(encoder, "encode_query", side_effect=EncoderUnavailable("offline")):
            result = selector.select(profile(), scope())
        self.assertEqual(result["mode"], "lexical_fallback")
        self.assertEqual(result["trace"]["fallback_reason"], "encoder_unavailable")
        with patch.object(encoder, "encode_query", side_effect=KnowledgeError("wrong model space")):
            with self.assertRaises(KnowledgeError):
                selector.select(profile(), scope())


if __name__ == "__main__":
    unittest.main()
