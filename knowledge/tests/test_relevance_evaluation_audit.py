"""Frozen unlabeled splits and personalized ranking from the October audit."""

import copy
import math
import unittest

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.evaluation import (
    assert_disjoint_splits, evaluate_predictions, split_rows,
)


def pair(name, *, company="A", role="cfo", cluster=None, grade=None, origin="human"):
    row = {"pair_id": name, "company": {"company_id": company}, "role": role,
           "article": {"article_id": cluster or name, "cluster_id": cluster or name,
                       "text": "Original publisher article " + (cluster or name)}}
    if grade is not None:
        row["label"] = {"grade": grade, "origin": origin,
                        "review_status": "agreed", "reviewers": ["alice", "bob"]}
    return row


class FrozenUnlabeledSplitTests(unittest.TestCase):
    def setUp(self):
        self.rows = [pair(f"story-{story}-{company}", company=company, cluster=f"story-{story}")
                     for story in range(8) for company in ("A", "B", "C")]

    def test_freeze_before_labeling_keeps_contrastive_company_triples_together(self):
        frozen = split_rows(self.rows, require_labels=False, seed=31)
        assert_disjoint_splits(frozen["train"], frozen["validation"], frozen["test"])
        membership = {row["pair_id"]: name for name in ("train", "validation", "test")
                      for row in frozen[name]}
        for story in range(8):
            self.assertEqual(len({membership[f"story-{story}-{company}"]
                                  for company in ("A", "B", "C")}), 1)
        for name in ("train", "validation", "test"):
            counts = frozen["label_counts"][name]
            self.assertEqual(counts["labeled"], 0)
            self.assertEqual(counts["unlabeled"], len(frozen[name]))
            self.assertIsNone(counts["positive_4_5"])
            self.assertIsNone(counts["negative_0_3"])
            self.assertIn(name + "_unlabeled_class_distribution_unknown", frozen["warnings"])
            self.assertNotIn(name + "_contains_only_one_class", frozen["warnings"])
        self.assertTrue(all("label" not in row for row in self.rows))

    def test_labels_do_not_change_frozen_memberships_or_fingerprint(self):
        frozen = split_rows(self.rows, require_labels=False, seed=31)
        labeled = copy.deepcopy(self.rows)
        for index, row in enumerate(labeled):
            row["label"] = {"grade": 5 if index % 2 else 0, "origin": "teacher"}
        after_labeling = split_rows(labeled, seed=31)
        self.assertEqual(frozen["split_fingerprint"], after_labeling["split_fingerprint"])
        self.assertTrue(after_labeling["labels_required"])

    def test_unlabeled_mode_does_not_relax_invalid_present_labels(self):
        with self.assertRaisesRegex(KnowledgeError, "labeled"):
            split_rows(self.rows)
        self.rows[0]["label"] = {"grade": True, "origin": "teacher"}
        with self.assertRaisesRegex(KnowledgeError, "integer"):
            split_rows(self.rows, require_labels=False)
        with self.assertRaisesRegex(KnowledgeError, "boolean"):
            split_rows([], require_labels="false")

    def test_company_holdout_still_refuses_connected_unlabeled_triples(self):
        with self.assertRaisesRegex(KnowledgeError, "independent"):
            split_rows(self.rows, mode="company", require_labels=False)
        separate = [pair(f"independent-{i}", company=f"company-{i}") for i in range(8)]
        frozen = split_rows(separate, mode="company", require_labels=False)
        assert_disjoint_splits(frozen["train"], frozen["validation"], frozen["test"], company_holdout=True)


class PersonalizedRankingTests(unittest.TestCase):
    def test_assisted_or_duplicate_reviewer_claims_are_not_human_gold(self):
        for changes in ({"independently_reviewed": False}, {"reviewers": ["alice", " alice "]}):
            row = pair("unverified", grade=4)
            row["label"].update(changes)
            self.assertFalse(evaluate_predictions([row], [0.9])["human_benchmark_eligible"])

    def test_ndcg_uses_full_business_grades_and_precision_uses_available_candidates(self):
        rows = [pair("noise", grade=0), pair("critical", grade=5), pair("useful", grade=4)]
        report = evaluate_predictions(rows, [0.99, 0.9, 0.8], budget_k=1)
        expected = (31 / math.log2(3) + 15 / math.log2(4)) / (31 + 15 / math.log2(3))
        self.assertAlmostEqual(report["ranking"]["ndcg_at_5"], expected)
        self.assertAlmostEqual(report["ranking"]["precision_at_5"], 2 / 3)
        self.assertEqual(report["ranking"]["selected_at_5"], 3)
        self.assertEqual(report["ranking"]["precision_at_k"], 0)
        self.assertTrue(report["human_benchmark_eligible"])

    def test_personalized_company_role_groups_are_not_ranked_against_each_other(self):
        rows = [pair("ac1", grade=5), pair("ac2", grade=0),
                pair("ae1", grade=4, role="ceo"), pair("ae2", grade=0, role="ceo"),
                pair("bc1", grade=0, company="B"), pair("bc2", grade=0, company="B")]
        report = evaluate_predictions(rows, [0.9, 0.1, 0.1, 0.9, 0.99, 0.98])
        groups = report["by_company_role"]
        self.assertEqual(groups["A"]["cfo"]["ranking"]["ndcg_at_5"], 1)
        self.assertAlmostEqual(groups["A"]["ceo"]["ranking"]["ndcg_at_5"], 1 / math.log2(3))
        self.assertIsNone(groups["B"]["cfo"]["ranking"]["ndcg_at_5"])
        macro = report["ranking_by_company_role_macro"]
        self.assertEqual(macro["group_count"], 3)
        self.assertEqual(macro["ndcg_at_5_defined_groups"], 2)
        self.assertAlmostEqual(macro["ndcg_at_5"], (1 + 1 / math.log2(3)) / 2)
        self.assertAlmostEqual(macro["precision_at_5"], 1 / 3)
        self.assertEqual(report["by_role"]["cfo"]["count"], 4)
        self.assertEqual(report["by_company"]["A"]["count"], 4)

    def test_zero_gain_empty_and_teacher_rankings_are_qualified(self):
        report = evaluate_predictions([pair("zero", grade=0, origin="teacher")], [0.8])
        self.assertIsNone(report["ranking"]["ndcg_at_5"])
        self.assertFalse(report["human_benchmark_eligible"])
        self.assertEqual(report["ranking"]["precision_at_5"], 0)
        empty = evaluate_predictions([], [])
        self.assertIsNone(empty["ranking"]["ndcg_at_5"])
        self.assertIsNone(empty["ranking"]["precision_at_5"])
        self.assertEqual(empty["ranking_by_company_role_macro"]["group_count"], 0)


if __name__ == "__main__":
    unittest.main()
