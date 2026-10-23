"""Local-only request fencing and immutable partial progress; no model calls."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.dataset import read_dataset, write_dataset
from alphich_kb.relevance_ml.local_qwen import export_batch, label_batch, local_endpoint
from test_relevance_teacher import judgement, pair


class LocalQwenTests(unittest.TestCase):
    def test_legacy_and_enriched_prompt_contracts_cannot_share_a_batch_or_make_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "pairs.jsonl"; output = root / "batch"
            enriched = pair(); enriched.pop("pair_id")
            enriched["company"]["facts"].append({"id": "staff-fns", "name": "employees_fns_band",
                                                  "value": "101-250", "source_ref": "synthetic fixture"})
            write_dataset(source, [pair(), enriched])
            with patch("alphich_kb.relevance_ml.local_qwen._request") as request:
                with self.assertRaisesRegex(KnowledgeError, "one prompt contract"):
                    label_batch(source, output, model="qwen-local", base_url="http://127.0.0.1:8000/v1")
            request.assert_not_called()
            self.assertFalse(output.exists())

    def test_only_explicit_loopback_without_credentials_queries_or_redirect_target(self):
        self.assertEqual(local_endpoint("http://localhost:8000/v1/"),
                         "http://127.0.0.1:8000/v1/chat/completions")
        self.assertEqual(local_endpoint("http://[::1]:8000"),
                         "http://[::1]:8000/v1/chat/completions")
        for url in ("https://api.example.com/v1", "http://127.0.0.1.evil:8000/v1",
                    "http://user@127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1?",
                    "http://127.0.0.1:8000/v1#", "http://127.0.0.1/v1",
                    "http://127.0.0.1:8000/other"):
            with self.subTest(url=url), self.assertRaises(KnowledgeError):
                local_endpoint(url)

    def test_valid_predictions_resume_export_and_do_not_claim_human_review(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "pairs.jsonl"; out = root / "batch"
            write_dataset(source, [pair()])
            proposed = judgement(); proposed["company_fact_ids"] = ["f1"]
            with patch("alphich_kb.relevance_ml.local_qwen._request", return_value=proposed) as request:
                first = label_batch(source, out, model="qwen-local", base_url="http://localhost:8000/v1")
                second = label_batch(source, out, model="qwen-local", base_url="http://localhost:8000/v1")
            self.assertEqual(request.call_count, 1)
            self.assertEqual(first, second)
            export_batch(out, root / "predictions.jsonl")
            row = read_dataset(root / "predictions.jsonl")[0]
            self.assertEqual(row["label"]["labeler"], "local:qwen-local")
            self.assertIs(row["label"]["independently_reviewed"], False)
            with self.assertRaises(KnowledgeError):
                label_batch(source, out, model="different", base_url="http://localhost:8000/v1")

    def test_ambiguous_failure_is_unlabeled_and_never_automatically_retried(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "pairs.jsonl"; out = root / "batch"
            write_dataset(source, [pair()])
            with patch("alphich_kb.relevance_ml.local_qwen._request", side_effect=KnowledgeError("timeout")) as request:
                for _ in range(2):
                    with self.assertRaises(KnowledgeError):
                        label_batch(source, out, model="qwen-local", base_url="http://127.0.0.1:8000/v1")
            self.assertEqual(request.call_count, 1)
            with self.assertRaises(KnowledgeError):
                export_batch(out, root / "empty.jsonl")
            self.assertFalse((root / "empty.jsonl").exists())

    def test_existing_human_labels_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root / "pairs.jsonl"
            row = pair(); row["label"] = {"origin": "human", "grade": 2}
            write_dataset(source, [row])
            with patch("alphich_kb.relevance_ml.local_qwen._request") as request:
                with self.assertRaises(KnowledgeError):
                    label_batch(source, root / "batch", model="qwen-local", base_url="http://127.0.0.1:8000/v1")
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
