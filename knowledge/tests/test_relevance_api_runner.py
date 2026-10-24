"""API teacher pilot accounting and crash/resume boundaries, without API calls."""
import copy
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from alphich_kb.common import KnowledgeError, read_json
from alphich_kb.relevance_ml.api_runner import batch_plan, run_api_batch
from alphich_kb.relevance_ml.dataset import validate_pair, write_dataset
from alphich_kb.relevance_ml.teacher import prepare_judgement, teacher_run_id
from app.intelligence.settings import AnalysisError, Model


MODEL = "z-ai/glm-5.3-flash"


def pair(index=1):
    return validate_pair({"company": {"company_id": "transport", "profile_version": "verified-v1", "facts": [
        {"id": "okved", "name": "okved_code", "value": "49.41", "source_ref": "synthetic verified fixture"}]},
        "article": {"article_id": str(index), "source_version": "original-1", "title": "Требования к перевозке",
                    "text": "Новые федеральные требования к перевозке грузов вступили в силу.",
                    "url": "https://example.org/news/" + str(index), "published_at": "2026-10-20T08:00:00Z"},
        "as_of": "2026-10-22T08:00:00Z", "role": "cfo"})


def judgement():
    return {"grade": 4, "impact_type": "regulation", "reason": "Требования применимы к подтверждённым перевозкам.",
            "news_quotes": ["Новые федеральные требования к перевозке грузов вступили в силу."],
            "company_fact_ids": ["okved"], "geography": "applicable", "scale": "material",
            "applicability": "confirmed", "event_status": "effective", "required_action": "Проверить документы",
            "missing_information": [], "timeliness": "unknown", "temporal_reason": "Актуальность не подтверждена реестром.",
            "temporal_quotes": [], "valid_from": None, "valid_until": None}


class LedgerProvider:
    def __init__(self, outcomes=()):
        self.settings = SimpleNamespace(models={MODEL: Model(MODEL, "glm-5.3-flash")}, analysis_model=MODEL)
        self.outcomes = list(outcomes)
        self.dispatched = []
        self.calls = {}

    def ledger(self, run_id):
        return copy.deepcopy(self.calls.get(run_id, []))

    def generate(self, run_id, *args):
        self.dispatched.append(run_id)
        outcome = self.outcomes.pop(0) if self.outcomes else "ok"
        if outcome in {"budget_exceeded", "gateway_daily_budget_not_verified"}:
            raise AnalysisError(outcome)
        unknown = outcome == "llm_timeout_outcome_unknown"
        http_rejection = outcome == "llm_http_503"
        self.calls[run_id] = [{"call_id": "llm_" + run_id, "run_id": run_id, "day": "2026-10-24", "model_id": MODEL,
            "status": "failed" if unknown or http_rejection else "completed", "request_hash": "fixture-hash",
            "cost_micros": None if unknown else 0 if http_rejection else 1_500_000,
            "input_tokens": None if unknown or http_rejection else 1500, "output_tokens": None if unknown or http_rejection else 500,
            "outstanding_reserve_micros": 500_000_000 if unknown else 0,
            "reserved_micros": 500_000_000, "error": outcome if unknown or http_rejection else None,
            "receipt": None if unknown else {"amount": "0" if http_rejection else "1.5", "rub_per_unit": "1", "source": "x-litellm-response-cost"}}]
        if unknown or http_rejection:
            raise AnalysisError(outcome)
        return {"grade": 0} if outcome == "invalid" else judgement()


class ApiRunnerTests(unittest.TestCase):
    def run_batch(self, rows, provider, destination, **kwargs):
        return run_api_batch(rows, provider, destination=destination, batch_id="pilot-immutable", model_id=MODEL,
                             ledger_reader=provider.ledger, **kwargs)

    def test_receipts_and_validation_cost_are_persisted_and_resume_is_free(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider()
            result = self.run_batch([pair(1), pair(2)], provider, folder)
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["known_cost_rub"], "3")
            self.assertEqual(result["cost_per_accepted_label_rub"], "1.5")
            self.assertEqual(result["input_tokens"], 3000)
            self.assertEqual(result["output_tokens"], 1000)
            self.assertFalse(read_json(Path(folder) / (pair()["pair_id"] + ".json"))["label"]["independently_reviewed"])
            receipt = read_json(Path(folder) / "audit" / (pair()["pair_id"] + ".json"))
            self.assertEqual(receipt["calls"][0]["receipt"]["amount"], "1.5")
            self.assertEqual(Path(result["audit_summary"]).stat().st_mode & 0o777, 0o600)
            self.run_batch([pair(1), pair(2)], provider, folder)
            self.assertEqual(len(provider.dispatched), 2)

    def test_invalid_answer_is_not_zero_and_no_fresh_paid_retry_is_created(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider(["invalid"])
            result = self.run_batch([pair()], provider, folder)
            self.assertEqual(result["completed"], 0)
            self.assertEqual(result["unlabeled_failures"], 1)
            self.assertEqual(result["known_cost_rub"], "1.5")
            self.assertIsNone(result["cost_per_accepted_label_rub"])
            self.assertFalse((Path(folder) / (pair()["pair_id"] + ".json")).exists())
            self.run_batch([pair()], provider, folder)
            self.assertEqual(len(provider.dispatched), 1)

    def test_timeout_keeps_reserve_and_blocks_remaining_batch_on_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider(["llm_timeout_outcome_unknown"])
            result = self.run_batch([pair(1), pair(2)], provider, folder)
            self.assertEqual(result["unknown_cost_calls"], 1)
            self.assertEqual(result["outstanding_reserve_micros"], 500_000_000)
            resumed = self.run_batch([pair(1), pair(2)], provider, folder)
            self.assertEqual(resumed["stop_reason"], "unknown_outcome_requires_manual_reconciliation")
            self.assertEqual(len(provider.dispatched), 1)

    def test_budget_rejection_can_resume_only_with_original_identity(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider(["budget_exceeded", "ok"])
            result = self.run_batch([pair()], provider, folder)
            self.assertEqual(result["pending"], 1)
            self.assertEqual(result["accounted_calls"], 0)
            self.assertFalse((Path(folder) / (pair()["pair_id"] + ".error.json")).exists())
            self.assertEqual(self.run_batch([pair()], provider, folder)["completed"], 1)
            self.assertEqual(provider.dispatched[0], provider.dispatched[1])

    def test_zero_receipt_503_remains_unlabeled_and_is_never_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider(["llm_http_503"])
            result = self.run_batch([pair()], provider, folder)
            self.assertEqual(result["completed"], 0)
            self.assertEqual(result["stop_reason"], "llm_http_503")
            self.assertEqual(result["known_cost_rub"], "0")
            self.assertEqual(result["usage_incomplete_calls"], 1)
            self.run_batch([pair()], provider, folder)
            self.assertEqual(len(provider.dispatched), 1)

    def test_read_only_budget_check_failure_can_resume_without_a_false_paid_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider(["gateway_daily_budget_not_verified", "ok"])
            first = self.run_batch([pair()], provider, folder)
            self.assertEqual(first["stop_reason"], "gateway_daily_budget_not_verified")
            self.assertEqual(first["unaccounted_attempts"], 0)
            self.assertEqual(first["unlabeled_failures"], 0)
            self.assertEqual(first["pending"], 1)
            second = self.run_batch([pair()], provider, folder)
            self.assertEqual(second["completed"], 1)
            self.assertEqual(provider.dispatched[0], provider.dispatched[1])

    def test_runtime_limit_stops_before_first_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider()
            times = iter((0, 2, 2))
            result = self.run_batch([pair()], provider, folder, max_seconds=1, clock=lambda: next(times))
            self.assertEqual(result["stop_reason"], "runtime_stop_before_dispatch")
            self.assertEqual(provider.dispatched, [])

    def test_soft_cost_stop_is_cumulative_not_a_second_daily_ledger(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider()
            result = self.run_batch([pair(1), pair(2)], provider, folder, max_cost_rub="1")
            self.assertEqual(result["completed"], 1)
            self.assertEqual(result["known_cost_rub"], "1.5")
            self.assertEqual(result["stop_reason"], "between_call_cost_stop")
            self.assertIn("next receipt can exceed", result["cost_stop_semantics"])
            self.run_batch([pair(1), pair(2)], provider, folder, max_cost_rub="1")
            self.assertEqual(len(provider.dispatched), 1)

    def test_wire_preflight_runs_for_every_row_before_dispatch(self):
        row = pair(2)
        row.pop("pair_id")
        row["article"]["text"] = "Н" * 20000
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider()
            with self.assertRaisesRegex(KnowledgeError, "input bound"):
                self.run_batch([pair(), validate_pair(row)], provider, folder)
            self.assertEqual(provider.dispatched, [])

    def test_human_gold_and_changed_batch_inputs_are_never_overwritten(self):
        row = pair()
        row["label"] = {"grade": 4, "origin": "human"}
        with self.assertRaisesRegex(KnowledgeError, "human-reviewed"):
            batch_plan([row], batch_id="pilot", model_id=MODEL)
        with tempfile.TemporaryDirectory() as folder:
            provider = LedgerProvider()
            self.run_batch([pair()], provider, folder)
            with self.assertRaisesRegex(KnowledgeError, "different immutable"):
                self.run_batch([pair(2)], provider, folder)
            self.assertEqual(len(provider.dispatched), 1)

    def test_cli_dry_run_needs_no_live_flag_key_or_database_and_never_constructs_provider(self):
        from alphich_kb.relevance_ml.cli import _judge
        with tempfile.TemporaryDirectory() as folder:
            input_path = Path(folder) / "pairs.jsonl"
            write_dataset(input_path, [pair()])
            provider = LedgerProvider()
            args = Namespace(input=str(input_path), output=str(Path(folder) / "dry"), batch_id="dry-pilot",
                             model_id=MODEL, live=False, shared_ledger=False, dry_run=True, env_file=None, limit=1)
            with patch("app.intelligence.settings.Settings.from_env", return_value=provider.settings), \
                    patch("app.intelligence.provider.Provider", side_effect=AssertionError("must not dispatch")), \
                    patch("app.core.make_session_factory", side_effect=AssertionError("must not touch ledger")):
                result = _judge(args)
            self.assertEqual(result["paid_calls"], 0)
            self.assertTrue(result["wire_bound_checked"])
            self.assertFalse(result["shared_ledger_touched"])

    def test_cli_offset_reaches_next_slice_without_relabeling_first_rows(self):
        from alphich_kb.relevance_ml.cli import _judge
        with tempfile.TemporaryDirectory() as folder:
            input_path = Path(folder) / "pairs.jsonl"
            write_dataset(input_path, [pair(1), pair(2), pair(3)])
            provider = LedgerProvider()
            args = Namespace(input=str(input_path), output=str(Path(folder) / "next-slice"),
                             batch_id="pilot-next-slice", model_id=MODEL, live=False, shared_ledger=False,
                             dry_run=True, env_file=None, limit=1, offset=1)
            with patch("app.intelligence.settings.Settings.from_env", return_value=provider.settings), \
                    patch("app.intelligence.provider.Provider", side_effect=AssertionError("must not dispatch")), \
                    patch("app.core.make_session_factory", side_effect=AssertionError("must not touch ledger")):
                result = _judge(args)
            preflight = read_json(result["preflight"])
            self.assertEqual(result["input_rows_skipped"], 1)
            self.assertEqual(preflight["pairs"][0]["pair_id"], pair(2)["pair_id"])
            self.assertEqual(len(preflight["requests"]), 1)

    def test_frozen_evaluation_rejects_teacher_benchmark_before_embedding(self):
        from alphich_kb.relevance_ml.cli import _train
        with tempfile.TemporaryDirectory() as folder:
            paths = {}
            for index, split in enumerate(("train", "validation", "test"), 1):
                row = pair(index)
                row["article"]["text"] += str(index)
                row["article"].pop("cluster_id")
                row.pop("pair_id")
                row = validate_pair(row)
                row["label"] = {"grade": 4, "origin": "teacher"}
                paths[split] = str(Path(folder) / (split + ".jsonl"))
                write_dataset(paths[split], [row])
            args = Namespace(input=paths["train"], validation_input=paths["validation"], test_input=paths["test"], split="article")
            with patch("alphich_kb.embeddings.E5Encoder", side_effect=AssertionError("must not embed")):
                with self.assertRaisesRegex(KnowledgeError, "independently human-reviewed"):
                    _train(args)


if __name__ == "__main__":
    unittest.main()
