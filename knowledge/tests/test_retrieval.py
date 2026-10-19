import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from alphich_kb.common import KnowledgeError, digest, read_jsonl, seal, write_jsonl
from alphich_kb.context import context_for_model

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from alphich_kb.embeddings import E5Encoder, EncoderUnavailable, mean_pool, normalize
    from alphich_kb.retriever import BankRetriever, RetrievalScope, collapse_pages, reciprocal_rank_fusion
    from alphich_kb.index import build_index


class FakeEncoder:
    space_id = "test_space"

    def encode_query(self, query):
        return np.array([1, 0, 0], dtype=np.float32)


class FailedEncoder(FakeEncoder):
    def encode_query(self, query):
        raise EncoderUnavailable("Simulated inference outage")


def make_index(root):
    rows = [
        ("old", "2026-07-31", "Прогноз ключевой ставки 13,5% на конец 2026 года.", [0.9, 0.1, 0]),
        ("new", "2026-09-11", "Прогноз ставки 14% на конец 2026 года и 12% на конец 2027 года.", [0.8, 0.2, 0]),
        ("future", "2027-01-01", "Будущий прогноз ставки.", [1, 0, 0]),
        ("unknown", None, "Недатированный прогноз ставки.", [1, 0, 0]),
        ("labor", "2026-08-01", "Кадры и зарплаты. Нехватка сотрудников ограничивает производство.", [0, 1, 0]),
    ]
    docs, items, pages, vectors = [], [], [], []
    for name, published, text, vector in rows:
        doc = {"id": name, "published_at": published}
        page = {"id": f"page_{name}", "document_id": name, "page": 1, "text": text,
                "text_version": digest(text), "quality_flags": ["ocr_unreviewed"]}
        item = {"id": name, "document_id": name, "published_at": published, "parent_id": page["id"],
                "page": 1, "title": name, "url": f"https://example.test/{name}", "text": text,
                "source_version": name, "content_hash": digest(name), "text_version": digest(text),
                "topics": ["labor" if name == "labor" else "policy_rate"], "ocr_checked": name in {"old", "new"},
                "quality_flags": [], "source_span": {"start": 0, "end": len(text)},
                "source_refs": [{"document_id": name, "page": 1, "start": 0, "end": len(text)}]}
        docs.append(doc); pages.append(page); items.append(item); vectors.append(vector)
    claims = []
    for name, doc, value, horizon, role, previous in [
        ("claim_old", "old", "13.5", "2026-12-31", "current_in_document", None),
        ("claim_new", "new", "14", "2026-12-31", "current_in_document", "claim_old"),
        ("claim_2027", "new", "12", "2027-12-31", "current_in_document", None),
        ("claim_previous", "new", "13.5", "2026-12-31", "previous_view", None),
    ]:
        claims.append({"id": name, "document_id": doc, "document_chunk_id": doc,
                       "published_at": next(d["published_at"] for d in docs if d["id"] == doc),
                       "metric": "policy_rate", "forecast_horizon": horizon, "kind": "forecast",
                       "position_role": role, "attributed_to": "alfa_bank", "geography": "RU", "scenario": "base",
                       "unit": "percent", "value": value, "revision_of": previous, "verification_status": "visually_verified"})
    write_jsonl(root / "documents.jsonl", docs)
    write_jsonl(root / "pages.jsonl", pages)
    write_jsonl(root / "items.jsonl", items)
    write_jsonl(root / "claims.jsonl", claims)
    np.save(root / "vectors.npy", normalize(vectors), allow_pickle=False)
    seal(root, {"kind": "bank_retrieval_index", "version": "index_test", "corpus_version": "corpus_test",
                "embedding_space_id": "test_space", "dimension": 3})
    return frozenset(d["id"] for d in docs)


@unittest.skipIf(np is None, "Install knowledge/requirements-rag.lock for retrieval tests")
class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.allowed = make_index(self.root)
        self.retriever = BankRetriever(self.root, FakeEncoder())

    def tearDown(self):
        self.tmp.cleanup()

    def scope(self, when="2026-10-20", allowed=None):
        return RetrievalScope(when, self.allowed if allowed is None else frozenset(allowed))

    def test_mean_pool_ignores_padding_before_l2_normalization(self):
        hidden = np.array([[[1, 2], [3, 4], [1000, 1000]]], dtype=np.float32)
        actual = mean_pool(hidden, [[1, 1, 0]])
        np.testing.assert_allclose(actual, normalize([[2, 3]]))
        with self.assertRaises(KnowledgeError):
            normalize([[0, 0]])

    def test_encoder_rejects_duplicate_prefix_and_overflow_before_inference(self):
        encoder = E5Encoder.__new__(E5Encoder)
        encoder.max_tokens = 4
        encoder.tokenizer = SimpleNamespace(encode_batch=lambda *a, **k: [SimpleNamespace(ids=[1] * 5)])
        with self.assertRaises(KnowledgeError):
            encoder._encode(["passage: passage: duplicated"], "passage: ")
        with self.assertRaisesRegex(KnowledgeError, "instead of truncating"):
            encoder._encode(["passage: long input"], "passage: ")

    def test_future_undated_and_forbidden_vectors_are_filtered_before_top_k(self):
        result = self.retriever.retrieve("ставка", self.scope(allowed={"old"}), mode="dense", limit=1)
        self.assertEqual([h["document_id"] for h in result["hits"]], ["old"])
        self.assertEqual(result["trace"]["eligible_items"], 1)
        before_new = self.retriever.retrieve("ставка", self.scope("2026-08-01"), mode="dense")
        self.assertNotIn("new", [h["document_id"] for h in before_new["hits"]])
        self.assertNotIn("unknown", [h["document_id"] for h in before_new["hits"]])
        self.assertNotIn("future", [h["document_id"] for h in before_new["hits"]])

    def test_empty_permission_scope_returns_no_context_or_claims(self):
        result = self.retriever.retrieve("ставка", self.scope(allowed=set()), metric="policy_rate", horizon="2026-12-31")
        self.assertEqual(result["status"], "not_found")
        self.assertEqual(result["hits"] + result["evidence"] + result["bank_claims"], [])

    def test_topic_filter_applies_to_both_rankers(self):
        for mode in ("lexical", "dense", "hybrid"):
            result = self.retriever.retrieve("зарплаты", self.scope(), mode=mode, topics=["labor"])
            self.assertEqual([h["document_id"] for h in result["hits"]], ["labor"])

    def test_rrf_uses_ranks_and_collapses_overlapping_page_hits(self):
        items = [{"parent_id": "page_a"}, {"parent_id": "page_a"}, {"parent_id": "page_b"}]
        ranked = collapse_pages([(0, 1000), (1, 999), (2, -5)], items, 10)
        self.assertEqual([r["rank"] for r in ranked], [1, 2])
        combined = reciprocal_rank_fusion({"lexical": ranked, "dense": [ranked[1]]})
        self.assertEqual(combined[0]["parent_id"], "page_b")
        self.assertAlmostEqual(combined[0]["rrf"], 2 / 62)

    def test_inference_outage_has_explicit_fallback_and_different_run_id(self):
        result = self.retriever.retrieve("ставка", self.scope())
        failed = BankRetriever(self.root, FailedEncoder())
        fallback = failed.retrieve("ставка", self.scope())
        self.assertEqual(fallback["mode"], "lexical_fallback")
        self.assertEqual(fallback["trace"]["fallback_reason"], "encoder_unavailable")
        self.assertNotEqual(result["retrieval_run_id"], fallback["retrieval_run_id"])
        with self.assertRaises(EncoderUnavailable):
            failed.retrieve("ставка", self.scope(), allow_fallback=False)

    def test_different_embedding_space_is_rejected(self):
        wrong = FakeEncoder()
        wrong.space_id = "different_model"
        with self.assertRaises(KnowledgeError):
            BankRetriever(self.root, wrong)

    def test_existing_index_is_rejected_before_encoding(self):
        encoder = SimpleNamespace()
        with self.assertRaisesRegex(KnowledgeError, "Output exists"):
            build_index(self.root, self.root, encoder)

    def test_inconsistent_dates_are_rejected_even_if_checksums_were_regenerated(self):
        items = read_jsonl(self.root / "items.jsonl")
        items[1]["published_at"] = "2020-01-01"
        write_jsonl(self.root / "items.jsonl", items)
        seal(self.root, {"kind": "bank_retrieval_index", "version": "bad", "corpus_version": "bad",
                         "embedding_space_id": "test_space", "dimension": 3})
        with self.assertRaisesRegex(KnowledgeError, "does not agree"):
            BankRetriever(self.root, FakeEncoder())

    def test_incompatible_forecasts_remain_conflicting(self):
        alternative = {**self.retriever.claims[1], "id": "independent_forecast", "value": "15", "revision_of": None}
        self.retriever.claims.append(alternative)
        result = self.retriever.retrieve("прогноз", self.scope(), metric="policy_rate", horizon="2026-12-31")
        self.assertEqual(result["bank_view_status"], "conflicting")
        self.assertEqual({c["value"] for c in result["bank_claims"]}, {"14", "15"})

    def test_registry_adds_revision_evidence_even_if_top_k_misses_its_page(self):
        result = self.retriever.retrieve("зарплаты", self.scope(), mode="lexical", limit=1,
                                         metric="policy_rate", horizon="2026-12-31")
        self.assertEqual([h["document_id"] for h in result["hits"]], ["labor"])
        self.assertEqual([c["value"] for c in result["bank_claims"]], ["14"])
        self.assertIn("new", [e["id"] for e in result["evidence"]])

    def test_revision_selection_respects_horizon_time_and_access(self):
        for when, allowed, horizon, expected in [
            ("2026-08-01", self.allowed, "2026-12-31", "13.5"),
            ("2026-10-20", self.allowed, "2026-12-31", "14"),
            ("2026-10-20", self.allowed, "2027-12-31", "12"),
            ("2026-10-20", {"old"}, "2026-12-31", "13.5"),
        ]:
            with self.subTest(when=when, allowed=allowed, horizon=horizon):
                result = self.retriever.retrieve("ставка", self.scope(when, allowed), metric="policy_rate", horizon=horizon)
                self.assertEqual([c["value"] for c in result["bank_claims"]], [expected])

    def test_unscoped_forecast_and_unknown_metric_do_not_invent_bank_view(self):
        for metric in (None, "missing_metric", "policy_rate"):
            result = self.retriever.retrieve("ставка", self.scope(), metric=metric)
            self.assertEqual(result["bank_claims"], [])
            self.assertEqual(result["bank_view_status"], "not_found")

    def test_expired_forecast_is_historical_only(self):
        result = self.retriever.retrieve("ставка", self.scope("2028-01-01"), metric="policy_rate", horizon="2026-12-31")
        self.assertEqual(result["bank_view_status"], "historical_only")
        self.assertIn("forecast_period_elapsed", result["limitations"])

    def test_expansion_has_own_citation_and_does_not_inherit_visual_review(self):
        result = self.retriever.retrieve("прогноз", self.scope(), mode="lexical", limit=1)
        context = result["parent_contexts"][0]
        self.assertTrue(context["id"].startswith("span_"))
        self.assertFalse(context["ocr_checked"])
        page = self.retriever.pages[context["parent_id"]]
        span = context["source_span"]
        self.assertEqual(context["text"], page["text"][span["start"]:span["end"]])
        self.assertEqual(self.retriever.get_evidence(context["id"], self.scope())["text"], context["text"])
        with self.assertRaises(KnowledgeError):
            self.retriever.get_evidence(context["id"], self.scope(allowed=set()))

    def test_generation_budget_counts_all_returned_source_text(self):
        result = self.retriever.retrieve("ставка", self.scope(), max_context_chars=512)
        context = context_for_model(result)
        size = sum(len(e["text"]) for e in context["untrusted_source_data"])
        self.assertLessEqual(size, 512)
        self.assertEqual(context["budget"]["used"], size)
        self.assertNotIn("hits", context)

    def test_model_input_and_returned_evidence_cannot_mutate_stored_sources(self):
        original = copy.deepcopy(self.retriever.by_id["old"])
        returned = self.retriever.get_evidence("old", self.scope())
        returned["source_refs"][0]["page"] = 999
        self.assertEqual(self.retriever.by_id["old"], original)


if __name__ == "__main__":
    unittest.main()
