import json
import tempfile
import unittest
from pathlib import Path

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.dataset import read_dataset
from alphich_kb.relevance_ml.pilot import (load_pilot, prepare_pilot, select_target_companies,
    validate_pilot_input, validate_pilot_splits)


def company(cid, amount="20000000000", unit="RUB", period="2025", code="41.20"):
    values = {"revenue_raw": amount, "revenue_unit": unit, "revenue_currency": unit,
              "revenue_period": period, "revenue_scope": "company_or_group_unknown", "okved_code": code}
    return {"company_id": cid, "profile_version": "source-v1", "facts": [
        {"id": cid + name, "name": name, "value": value, "source_ref": "test:csv:row:1"}
        for name, value in values.items()]}


def articles():
    return [{"article_id": f"a{i}", "source_version": "source-v1", "cluster_id": f"story{i // 2}",
        "title": f"Construction event {i}", "text": f"Source-bound construction news with distinct detail {i}.",
        "url": f"https://example.test/news/{i}", "published_at": "2026-10-20T00:00:00Z",
        "topics": ["construction"]} for i in range(40)]


class ApiPilotTests(unittest.TestCase):
    def test_revenue_is_strict_unit_bound_with_explicit_scope(self):
        rows = [company("eligible"), company("equal", "15000000000"), company("thousands", unit="THOUSAND_RUB"),
                company("placeholder", "1000"), company("old", period="2024")]
        cohort, report = select_target_companies(rows)
        self.assertEqual([row["company_id"] for row in cohort], ["eligible"])
        self.assertEqual(report["excluded_counts"]["revenue_at_or_below_threshold"], 1)
        self.assertEqual(next(f["value"] for f in cohort[0]["facts"] if f["name"] == "revenue_scope"),
                         "company_or_group_unknown")
        with self.assertRaises(KnowledgeError):
            select_target_companies(rows, required_companies=["equal"])

    def test_pilot_has_frozen_blind_splits_all_roles_and_no_labels(self):
        companies = [company("construction"), company("transport", code="49.41"), company("trade", code="47.11")]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pilot"
            result = prepare_pilot(articles(), companies, root, as_of="2026-10-24T00:00:00Z", limit=60)
            manifest, splits = load_pilot(root / "manifest.json")
            self.assertEqual(result["paid_calls"], 0)
            self.assertEqual(sum(result["split_counts"].values()), 60)
            clusters = {}
            for name, rows in splits.items():
                self.assertTrue(rows)
                for row in rows:
                    self.assertNotIn("label", row)
                    self.assertEqual(clusters.setdefault(row["article"]["cluster_id"], name), name)
                self.assertEqual({row["role"] for row in rows}, {"cfo", "ceo"})
            self.assertEqual(validate_pilot_input(root / "manifest.json", root / "train.jsonl")["plan_id"], manifest["plan_id"])
            coverage = validate_pilot_splits(root / "manifest.json", {name: values[:1] for name, values in splits.items()})
            self.assertFalse(coverage["coverage"]["train"]["complete"])
            moved = {name: list(values) for name, values in splits.items()}
            moved["train"].append(moved["test"].pop())
            with self.assertRaises(KnowledgeError):
                validate_pilot_splits(root / "manifest.json", moved)
            with self.assertRaises(KnowledgeError):
                validate_pilot_input(root / "manifest.json", root / "validation.jsonl", split="validation")
            with self.assertRaises(KnowledgeError):
                validate_pilot_input(root / "manifest.json", root / "test.jsonl")
            changed = read_dataset(root / "train.jsonl")
            changed[0]["company"]["facts"][0]["value"] = "changed"
            (root / "train.jsonl").write_text(json.dumps(changed[0]) + "\n")
            with self.assertRaises(KnowledgeError):
                load_pilot(root / "manifest.json")

    def test_freshness_and_required_company_do_not_infer_relevance(self):
        old = articles()
        for article in old:
            article["published_at"] = "2025-01-01T00:00:00Z"
        with tempfile.TemporaryDirectory() as tmp, self.assertRaises(KnowledgeError):
            prepare_pilot(old, [company("a"), company("b")], Path(tmp) / "pilot",
                          as_of="2026-10-24T00:00:00Z")


if __name__ == "__main__":
    unittest.main()
