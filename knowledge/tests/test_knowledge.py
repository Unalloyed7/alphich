import copy
import tempfile
import unittest
from pathlib import Path

from alphich_kb.build import chunk_spans, make_chunk, prepare_claims, validate_date
from alphich_kb.common import KnowledgeError, artifact_dir, digest, seal, verify_artifacts, write_jsonl
from alphich_kb.extract import parse_tsv
from alphich_kb.search import available, search, select_claims


class CharacterCounter:
    """Deliberately strict counter for adversarial Unicode/boundary tests."""
    def count(self, text):
        return len(text) + 2


def fixture():
    doc = {"id": "rates", "version": "v1", "title": "Ставка", "published_at": "2026-09-11",
           "publication_precision": "day", "content_hash": "pdfhash", "url": "https://example.test/report",
           "topics": ["policy_rate"]}
    text = "Мы повысили прогноз до 14%. Прежний прогноз составлял 13,5%."
    page = {"id": "page_v1", "document_id": "rates", "page": 1, "text": text, "text_version": "ocrhash",
            "lines": [{"start": 0, "end": len(text), "bbox": [10, 20, 100, 30]}],
            "quality_flags": ["ocr_unreviewed", "layout_requires_review"], "image": "pages/rates/p001.png"}
    review = {"id": "new2026", "document_id": "rates", "page": 1, "pdf_sha256": "pdfhash",
              "page_text_version": "ocrhash", "quote": "Мы повысили прогноз до 14%.",
              "statement": "Прогноз ставки на конец 2026 года повышен до 14%.",
              "kind": "forecast", "attributed_to": "alfa_bank", "metric": "policy_rate", "geography": "RU",
              "scenario": "base", "value": "14", "unit": "percent", "observed_period": None,
              "forecast_horizon": "2026-12-31", "position_role": "current_in_document", "revision_of": None,
              "verification_status": "visually_verified", "review_note": "Checked number, horizon and author in image"}
    return doc, page, review


class KnowledgeTests(unittest.TestCase):
    def test_tsv_preserves_paragraph_order_numbers_and_coordinates(self):
        header = "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext\n"
        rows = ["1\t1\t0\t0\t0\t0\t0\t0\t500\t800\t-1\t",
                "5\t1\t1\t1\t1\t1\t10\t20\t40\t10\t95\tСтавка",
                "5\t1\t1\t1\t1\t2\t60\t20\t30\t10\t50\t13,5%",
                "5\t1\t2\t1\t1\t1\t10\t60\t60\t10\t90\tПрогноз"]
        page = parse_tsv(header + "\n".join(rows))
        self.assertEqual(page["text"], "Ставка 13,5%\n\nПрогноз")
        self.assertEqual(page["low_confidence_words"], 1)
        self.assertEqual(page["lines"][0]["bbox"], [10, 20, 80, 10])
        for line in page["lines"]:
            self.assertTrue(page["text"][line["start"]:line["end"]])

    def test_invalid_tsv_fails_instead_of_treating_diagnostic_as_text(self):
        with self.assertRaises(KnowledgeError):
            parse_tsv("read_params_file: Can't open tsv")

    def test_chunker_preserves_all_nonwhitespace_and_exact_budget(self):
        text = ("Рост инфляции. Снижение ставки зависит от условий.\n\n" * 20) + "длинноеслово" * 20
        counter = CharacterCounter()
        spans = list(chunk_spans(text, "passage: Тест\n", counter, 100))
        self.assertGreater(len(spans), 10)
        self.assertTrue(all(counter.count("passage: Тест\n" + text[s:e]) <= 100 for s, e in spans))
        self.assertTrue(all(any(s <= i < e for s, e in spans) for i, c in enumerate(text) if not c.isspace()))
        self.assertTrue(all(left[0] < right[0] for left, right in zip(spans, spans[1:])))

    def test_empty_page_and_oversized_metadata(self):
        self.assertEqual(list(chunk_spans("  \n", "passage: ", CharacterCounter(), 100)), [])
        with self.assertRaises(KnowledgeError):
            list(chunk_spans("Текст", "x" * 100, CharacterCounter(), 100))

    def test_evidence_is_bound_to_exact_page_span_and_review_scope(self):
        doc, page, _ = fixture()
        chunk = make_chunk(doc, page, 0, 25, {}, CharacterCounter())
        self.assertEqual(chunk["text"], page["text"][:25])
        self.assertFalse(chunk["ocr_checked"])
        self.assertIsNone(chunk["forecast_horizon"])
        changed = copy.deepcopy(page)
        changed["text_version"] = "ocrhash2"
        self.assertNotEqual(chunk["id"], make_chunk(doc, changed, 0, 25, {}, CharacterCounter())["id"])

    def test_review_enables_only_exact_quote_not_full_page(self):
        doc, page, review = fixture()
        evidence, claims = prepare_claims({"claims": [review]}, [doc], [page], {}, CharacterCounter())
        self.assertEqual(evidence[0]["text"], review["quote"])
        self.assertTrue(evidence[0]["ocr_checked"])
        self.assertEqual(claims[0]["document_chunk_id"], evidence[0]["id"])
        self.assertNotIn("ocr_checked", page)

    def test_stale_or_nonexact_review_is_rejected(self):
        doc, page, review = fixture()
        for field, value in (("pdf_sha256", "wrong"), ("page_text_version", "wrong"), ("quote", "14.0%"),
                             ("verification_status", "auto"), ("forecast_horizon", None)):
            invalid = {**review, field: value}
            with self.subTest(field=field), self.assertRaises(KnowledgeError):
                prepare_claims({"claims": [invalid]}, [doc], [page], {}, CharacterCounter())

    def test_revision_must_match_horizon_and_point_backwards(self):
        doc, page, review = fixture()
        earlier = {**doc, "id": "earlier", "published_at": "2026-07-31"}
        earlier_page = {**page, "document_id": "earlier"}
        old_review = {**review, "id": "old2026", "document_id": "earlier", "value": "13.5"}
        review["revision_of"] = "old2026"
        _, claims = prepare_claims({"claims": [old_review, review]}, [earlier, doc], [earlier_page, page], {}, CharacterCounter())
        self.assertEqual(claims[1]["revision_of"], claims[0]["id"])
        old_review["forecast_horizon"] = "2027-12-31"
        with self.assertRaises(KnowledgeError):
            prepare_claims({"claims": [old_review, review]}, [earlier, doc], [earlier_page, page], {}, CharacterCounter())

    def test_reviewed_nonfinancial_claim_needs_free_text_topic_instead_of_metric(self):
        doc, page, review = fixture()
        text = "Утечка данных может подорвать доверие клиентов к бренду."
        page = {**page, "text": text, "lines": [{"start": 0, "end": len(text), "bbox": [1, 2, 3, 4]}]}
        review = {**review, "kind": "mechanism", "metric": None, "topic": "Доверие клиентов и репутация",
                  "statement": text, "quote": text, "value": None, "unit": None, "forecast_horizon": None}
        evidence, claims = prepare_claims({"claims": [review]}, [doc], [page], {}, CharacterCounter())
        self.assertIsNone(claims[0]["metric"])
        self.assertEqual(claims[0]["topic"], "Доверие клиентов и репутация")
        self.assertEqual(evidence[0]["text"], text)
        for field, value in (("topic", None), ("topic", []), ("attributed_to", None), ("claim_key", "")):
            with self.subTest(field=field), self.assertRaises(KnowledgeError):
                prepare_claims({"claims": [{**review, field: value}]}, [doc], [page], {}, CharacterCounter())

    def test_qualitative_revision_requires_explicit_compatible_review(self):
        doc, page, review = fixture()
        review = {**review, "kind": "interpretation", "metric": None, "topic": "legal_obligations",
                  "claim_key": "specific_labeling_requirement", "value": None, "unit": None,
                  "forecast_horizon": None, "revision_of": "old"}
        old_doc = {**doc, "id": "old_doc", "published_at": "2026-07-31"}
        old_page = {**page, "document_id": old_doc["id"]}
        old = {**review, "id": "old", "document_id": old_doc["id"], "revision_of": None}
        _, claims = prepare_claims({"claims": [old, review]}, [old_doc, doc], [old_page, page], {}, CharacterCounter())
        self.assertEqual(claims[1]["revision_of"], claims[0]["id"])
        with self.assertRaisesRegex(KnowledgeError, "Revision changes topic"):
            prepare_claims({"claims": [old, {**review, "topic": "unrelated_legal_matter"}]},
                           [old_doc, doc], [old_page, page], {}, CharacterCounter())

    def test_time_filter_excludes_unknown_and_future_dates(self):
        self.assertFalse(available({"published_at": None}, "2026-10-20"))
        self.assertFalse(available({"published_at": "2026-09-11"}, "2026-09-10"))
        self.assertTrue(available({"published_at": "2026-09-11"}, "2026-09-11"))
        validate_date({"published_at": None, "publication_precision": "unknown"})

    def test_forecasts_keep_horizon_role_and_historical_snapshot(self):
        _, _, review = fixture()
        old = {**review, "id": "old", "published_at": "2026-07-31"}
        new = {**review, "id": "new", "published_at": "2026-09-11", "revision_of": "old"}
        other_horizon = {**new, "id": "2027", "forecast_horizon": "2027-12-31", "revision_of": None}
        previous_view = {**new, "id": "quoted_old", "position_role": "previous_view", "revision_of": None}
        data = [old, new, other_horizon, previous_view]
        self.assertEqual([c["id"] for c in select_claims(data, "2026-08-01", "policy_rate", "2026-12-31")], ["old"])
        self.assertEqual([c["id"] for c in select_claims(data, "2026-10-20", "policy_rate", "2026-12-31")], ["new"])
        self.assertEqual([c["id"] for c in select_claims(data, "2026-10-20", "policy_rate", "2027-12-31")], ["2027"])

    def test_artifacts_are_immutable_checked_and_atomic(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "snapshot"
            with artifact_dir(output) as stage:
                (stage / "a.jsonl").write_text("{}\n")
                seal(stage, {"version": digest("one")})
            verify_artifacts(output)
            with self.assertRaises(KnowledgeError):
                with artifact_dir(output):
                    pass
            (output / "a.jsonl").write_text("tampered")
            with self.assertRaises(KnowledgeError):
                verify_artifacts(output)
            failed = Path(tmp) / "failed"
            with self.assertRaises(RuntimeError):
                with artifact_dir(failed):
                    raise RuntimeError("OCR failed")
            self.assertFalse(failed.exists())

    def test_search_filters_future_unknown_and_topic_before_ranking(self):
        doc, page, _ = fixture()
        old = make_chunk({**doc, "published_at": "2026-07-31"}, page, 0, len(page["text"]), {}, CharacterCounter())
        future = {**old, "id": "future", "parent_id": "future_page", "published_at": "2027-01-01"}
        unknown = {**old, "id": "unknown", "parent_id": "unknown_page", "published_at": None}
        repeated = {**old, "id": "overlap"}
        with tempfile.TemporaryDirectory() as tmp:
            write_jsonl(Path(tmp) / "chunks.jsonl", [future, unknown, old, repeated])
            seal(Path(tmp), {"version": "fixture"})
            hits = search(tmp, "прогноз", "2026-08-01", limit=1)
            self.assertEqual([h["evidence_id"] for h in hits], [old["id"]])
            self.assertEqual(len(search(tmp, "прогноз", "2026-08-01", limit=10)), 1)
            self.assertEqual(search(tmp, "прогноз", "2026-08-01", topics=["gold"]), [])
            self.assertEqual(search(tmp, "прогноз", "2020-01-01"), [])

    def test_search_input_cannot_escape_fts_query_or_skip_invalid_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_jsonl(Path(tmp) / "chunks.jsonl", [])
            seal(Path(tmp), {"version": "fixture"})
            self.assertEqual(search(tmp, '" OR *; DROP TABLE fragments; --', "2026-08-01"), [])
            with self.assertRaises(ValueError):
                search(tmp, "ставка", "not-a-date")
            with self.assertRaises(ValueError):
                select_claims([], "not-a-date", "policy_rate", "2026-12-31")


if __name__ == "__main__":
    unittest.main()
