"""Source/label separation and invalid-judge behavior; no paid API calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.dataset import input_hash, read_dataset, validate_pair, write_dataset
from alphich_kb.relevance_ml.teacher import judge_pair, prepare_judgement, validate_checkpoint, validate_judgement


def pair():
    return validate_pair({
        "company": {"company_id": "EXAMPLE", "profile_version": "source-v1", "facts": [
            {"id": "industry-ref", "name": "industry", "value": {"code": "49.41", "name": "Грузоперевозки"},
             "source_ref": "synthetic fixture; not a real company"},
            {"id": "region-ref", "name": "registration_region", "value": "Москва", "source_ref": "fixture"}]},
        "article": {"article_id": "article-1", "source_version": "1", "title": "Изменение перевозочных документов",
                    "text": "Федеральные требования к перевозочным документам вступили в силу. Проверьте применимость.",
                    "url": "https://example.org/news/1", "published_at": "2026-10-20T08:00:00+00:00"},
        "as_of": "2026-10-22T08:00:00+00:00", "role": "cfo"})


def judgement():
    return {"grade": 5, "impact_type": "regulation", "reason": "Проверить документы по подтверждённой основной деятельности.",
            "news_quotes": ["Федеральные требования к перевозочным документам вступили в силу."],
            "company_fact_ids": ["industry-ref"], "geography": "applicable", "scale": "material",
            "applicability": "confirmed", "event_status": "effective", "required_action": "Проверить применимые документы",
            "missing_information": ["Проверить исключения для выполняемых перевозок"],
            "timeliness": "unknown", "temporal_reason": "Текст не подтверждает актуальность нормы на дату оценки.",
            "temporal_quotes": [], "valid_from": None, "valid_until": None}


class TeacherTests(unittest.TestCase):
    def test_preparation_never_contains_previous_annotation(self):
        row = pair()
        baseline = prepare_judgement(row)
        row["label"] = {"grade": 5, "origin": "teacher", "reason": "SECRET_LABEL_REASON"}
        row["teacher_grade"] = 5
        self.assertEqual(prepare_judgement(row), baseline)

    def test_foreign_quotes_and_company_ids_are_not_labels(self):
        for field, value in (("news_quotes", ["Invented source text"]), ("company_fact_ids", ["foreign-company"])):
            with self.subTest(field=field):
                response = judgement(); response[field] = value
                with self.assertRaises(KnowledgeError):
                    validate_judgement(response, pair())

    def test_high_grade_conflicts_fail_not_silently_become_zero(self):
        for field, value in (("geography", "unknown"), ("scale", "limited"), ("applicability", "conditional"),
                             ("event_status", "proposal"), ("required_action", None), ("grade", True)):
            with self.subTest(field=field):
                response = judgement(); response[field] = value
                with self.assertRaises(KnowledgeError):
                    validate_judgement(response, pair())

    def test_missing_date_and_registration_alone_cannot_support_five(self):
        row = pair(); row["article"]["published_at"] = None; row.pop("pair_id")
        response = judgement(); response["timeliness"] = "current"
        response["temporal_quotes"] = response["news_quotes"]
        with self.assertRaises(KnowledgeError):
            validate_judgement(response, row)
        response = judgement(); response["company_fact_ids"] = ["region-ref"]
        with self.assertRaises(KnowledgeError):
            validate_judgement(response, pair())

    def test_archive_business_grade_does_not_depend_on_publication_age(self):
        row = pair(); row.pop("pair_id")
        row["article"]["published_at"] = "2025-01-20T08:00:00+00:00"
        response = judgement(); response["event_status"] = "historical"
        self.assertEqual(validate_judgement(response, row)["grade"], 5)
        self.assertEqual(validate_judgement(response, row)["timeliness"], "unknown")

    def test_explicit_expiry_is_separate_from_business_fit_and_cannot_be_current(self):
        row = pair(); row.pop("pair_id")
        quote = "Требование действует с 2025-01-01 до 2025-12-31."
        row["article"]["text"] += " " + quote
        response = judgement()
        response.update(timeliness="expired", temporal_quotes=[quote], valid_from="2025-01-01", valid_until="2025-12-31")
        self.assertEqual(validate_judgement(response, row)["grade"], 5)
        response["timeliness"] = "current"
        with self.assertRaisesRegex(KnowledgeError, "validity interval"):
            validate_judgement(response, row)
        response["timeliness"] = "expired"; response["valid_until"] = "2024-12-31"
        with self.assertRaisesRegex(KnowledgeError, "absent from quoted"):
            validate_judgement(response, row)

    def test_invalid_teacher_result_keeps_original_row_unlabeled(self):
        class BadProvider:
            def generate(self, *args):
                return {"grade": 0}
        row = pair()
        with self.assertRaises(KnowledgeError):
            judge_pair(row, BadProvider(), batch_id="batch-1", model_id="allowed-model")
        self.assertNotIn("label", row)

    def test_checkpoint_cannot_change_model_grade_or_evidence(self):
        class Provider:
            def generate(self, *args):
                return judgement()
        row = judge_pair(pair(), Provider(), batch_id="batch-1", model_id="allowed-model")
        plan = prepare_judgement(row)
        batch = {"batch_id": "batch-1", "model_id": "allowed-model", "prompt_hash": plan["prompt_hash"],
                 "requests": [plan["input_hash"]]}
        self.assertEqual(validate_checkpoint(row, batch), row)
        for field, value in (("labeler", "foreign-model"), ("grade", 0), ("run_id", "other-call")):
            changed = copy.deepcopy(row); changed["label"][field] = value
            with self.assertRaises(KnowledgeError):
                validate_checkpoint(changed, batch)

    def test_shared_provider_call_identity_and_label_origin(self):
        calls = []
        class Provider:
            def generate(self, *args):
                calls.append(args)
                return judgement()
        row = pair()
        one = judge_pair(row, Provider(), batch_id="batch-1", model_id="allowed-model")
        two = judge_pair(row, Provider(), batch_id="batch-1", model_id="allowed-model")
        self.assertEqual(calls[0], calls[1])
        self.assertLessEqual(len(calls[0][0]), 40)
        self.assertEqual(one["label"]["origin"], "teacher")
        self.assertFalse(one["label"]["independently_reviewed"])
        self.assertEqual(one["label"]["run_id"], two["label"]["run_id"])

    def test_snapshot_change_invalidates_pair_and_bound_label(self):
        row = pair()
        row["label"] = {"grade": 4, "origin": "teacher", "input_hash": input_hash(row)}
        row["company"]["facts"][0]["value"]["code"] = "62.01"
        with self.assertRaises(KnowledgeError):
            validate_pair(row)
        row.pop("pair_id")
        with self.assertRaises(KnowledgeError):
            validate_pair(row)

    def test_private_create_only_roundtrip_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.json"
            write_dataset(path, [pair()])
            self.assertEqual(read_dataset(path), [pair()])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_dataset(path, [pair()])
            duplicate = Path(folder) / "duplicates.json"
            with self.assertRaises(KnowledgeError):
                write_dataset(duplicate, [pair(), pair()])
            self.assertFalse(duplicate.exists())
            duplicate.write_text(json.dumps([pair(), pair()]), encoding="utf-8")
            with self.assertRaises(KnowledgeError):
                read_dataset(duplicate)

    def test_annotation_in_a_company_fact_is_rejected(self):
        row = copy.deepcopy(pair()); row.pop("pair_id")
        row["company"]["facts"][0]["value"]["teacher_grade"] = 5
        with self.assertRaises(KnowledgeError):
            validate_pair(row)

    def test_paid_cli_uses_existing_shared_500_cap_without_resetting_spend_or_reserves(self):
        from argparse import Namespace
        from datetime import datetime, timezone
        from decimal import Decimal
        from unittest.mock import patch

        from app.core import make_session_factory
        from app.intelligence.settings import Model, Settings
        from app.intelligence.storage import BudgetDay
        from alphich_kb.relevance_ml.cli import _judge

        model_id = "z-ai/glm-5.3-flash"
        settings = Settings(base_url="https://example.org/v1", api_key="test-only",
                            extraction_model=model_id, analysis_model=model_id,
                            models={model_id: Model(model_id, "glm-5.3-flash")},
                            budget_mode="gateway_receipt", receipt_rub_per_unit=Decimal("1"),
                            daily_limit_micros=500_000_000)

        class OfflineProvider:
            def __init__(self, *args):
                pass

            def generate(self, *args):
                return judgement()

        # SQLite is only the isolated test fixture. Production must still use
        # the server's shared PostgreSQL URL and its existing accounting row.
        for existing_cap in (500_000_000, 500_000_001):
            with self.subTest(existing_cap=existing_cap), tempfile.TemporaryDirectory() as folder:
                engine, factory = make_session_factory("sqlite:///" + str(Path(folder) / "ledger.db"))
                BudgetDay.__table__.create(engine)
                day = datetime.now(timezone.utc).date().isoformat()
                with factory.begin() as db:
                    db.add(BudgetDay(day=day, limit_micros=existing_cap,
                                     spent_micros=120_000_000, reserved_micros=25_000_000))
                input_file = Path(folder) / "pairs.json"
                write_dataset(input_file, [pair()])
                args = Namespace(live=True, shared_ledger=True, env_file=None, limit=1,
                                 input=str(input_file), output=str(Path(folder) / "results"),
                                 model_id=model_id, batch_id="shared-cap-test")
                try:
                    with patch.dict("os.environ", {"DATABASE_URL": "postgresql://test-only/shared"}), \
                            patch("app.core.make_session_factory", return_value=(engine, factory)), \
                            patch("app.intelligence.settings.Settings.from_env", return_value=settings), \
                            patch("app.intelligence.provider.Provider", OfflineProvider):
                        if existing_cap > 500_000_000:
                            with self.assertRaisesRegex(KnowledgeError, "shared ledger cap exceeds 500"):
                                _judge(args)
                        else:
                            self.assertEqual(_judge(args)["completed"], 1)
                    with factory() as db:
                        ledger = db.get(BudgetDay, day)
                        self.assertEqual((ledger.limit_micros, ledger.spent_micros, ledger.reserved_micros),
                                         (existing_cap, 120_000_000, 25_000_000))
                finally:
                    engine.dispose()


if __name__ == "__main__":
    unittest.main()
