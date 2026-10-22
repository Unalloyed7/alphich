"""Narrow numerical, provenance and leakage checks for the new CPU baseline."""

import copy
import unittest

from alphich_kb.common import KnowledgeError, digest
from alphich_kb.relevance_ml.evaluation import (assert_disjoint_splits, calibrate_threshold,
                                               evaluate_predictions, split_rows)

try:
    import numpy as np
except ImportError:
    np = None

if np is not None:
    from alphich_kb.relevance_ml.model import predict, train_model, validate_artifact


def pair(name, grade, *, company="A", origin="human", cluster=None, text=None):
    return {"pair_id": name, "company": {"company_id": company, "profile_version": "1", "facts": []},
            "article": {"article_id": name, "cluster_id": cluster or name, "title": name,
                        "text": text or f"article text {name}"},
            "label": {"grade": grade, "origin": origin, "reviewers": ["alice", "bob"],
                      "review_status": "agreed"}}


MANIFEST = {"feature_version": "test-v1", "encoder_space_id": "e5-test", "dimension": 2,
            "feature_names": ["first", "second"]}


class RelevanceEvaluationTests(unittest.TestCase):
    def test_undefined_recall_and_ap_are_null(self):
        report = evaluate_predictions([pair("x", 0), pair("y", 3)], [0.1, 0.8])
        self.assertIsNone(report["recall"])
        self.assertIsNone(report["ranking"]["average_precision"])
        self.assertEqual(report["confusion"], {"tp": 0, "fp": 1, "fn": 0, "tn": 1})
        self.assertTrue(report["human_benchmark_eligible"])

    def test_teacher_cannot_be_human_benchmark_and_critical_misses_are_visible(self):
        report = evaluate_predictions([pair("x", 5, origin="teacher"), pair("y", 4), pair("z", 0)],
                                      [0.1, 0.9, 0.8], threshold=0.5, budget_k=2)
        self.assertEqual(report["missed_grade5_pair_ids"], ["x"])
        self.assertFalse(report["human_benchmark_eligible"])
        self.assertEqual(report["human_benchmark_eligible_count"], 2)
        self.assertEqual(report["ranking"]["precision_at_k"], 0.5)
        self.assertEqual(report["ranking"]["recall_at_budget"], 0.5)

    def test_average_precision_does_not_benefit_from_score_tie_order(self):
        a, b = pair("a", 5), pair("b", 0)
        first = evaluate_predictions([a, b], [0.5, 0.5])
        second = evaluate_predictions([b, a], [0.5, 0.5])
        self.assertEqual(first["ranking"]["average_precision"], 0.5)
        self.assertEqual(first["ranking"]["average_precision"], second["ranking"]["average_precision"])

    def test_article_pairs_and_exact_reprints_cannot_leak(self):
        rows = [pair(f"p{i}", 4 if i % 2 else 1, cluster=f"cluster{i // 2}") for i in range(12)]
        rows.extend([pair("copy1", 3, text="EXACT   same copy"), pair("copy2", 4, text="exact same COPY")])
        result = split_rows(rows, seed=3)
        assert_disjoint_splits(result["train"], result["validation"], result["test"])
        memberships = {row["pair_id"]: name for name in ("train", "validation", "test") for row in result[name]}
        self.assertEqual(memberships["copy1"], memberships["copy2"])
        for i in range(0, 12, 2):
            self.assertEqual(memberships[f"p{i}"], memberships[f"p{i+1}"])

    def test_company_holdout_refuses_a_fully_connected_dataset(self):
        rows = [pair(f"{company}{i}", 4 if i else 0, company=company, cluster=f"article{i}")
                for company in ("A", "B", "C") for i in range(3)]
        with self.assertRaisesRegex(KnowledgeError, "independent"):
            split_rows(rows, mode="company")

    def test_threshold_selection_is_human_validation_only_and_rejects_overlap(self):
        train = [pair("training", 4)]
        validation = [pair("v1", 5), pair("v2", 4), pair("v3", 0)]
        selection = calibrate_threshold(validation, [0.9, 0.4, 0.7], training_rows=train, min_recall=1)
        self.assertEqual(selection["threshold"], 0.4)
        self.assertEqual(selection["observed_recall"], 1)
        self.assertFalse(selection["production_calibrated"])
        self.assertLess(selection["recall_wilson_95_lower"], 0.5)
        validation[0]["label"]["origin"] = "teacher"
        with self.assertRaises(KnowledgeError):
            calibrate_threshold(validation, [0.9, 0.4, 0.7], training_rows=train)
        with self.assertRaisesRegex(KnowledgeError, "leakage"):
            assert_disjoint_splits(train, [copy.deepcopy(train[0])])


@unittest.skipIf(np is None, "Install knowledge/requirements-rag.lock for CPU model tests")
class RelevanceModelTests(unittest.TestCase):
    def setUp(self):
        self.rows = [pair("a", 0), pair("b", 2), pair("c", 4), pair("d", 5)]
        self.values = np.array([[-2, 1], [-1, 1], [1, 1], [2, 1]], dtype=np.float64)

    def test_legacy_freshness_capped_teacher_labels_require_relabeling(self):
        rows = copy.deepcopy(self.rows)
        for row in rows:
            row["label"].update(origin="teacher", label_version="relevance-judge-v1")
        with self.assertRaisesRegex(KnowledgeError, "need relabeling"):
            train_model(rows, self.values, feature_manifest=MANIFEST, allow_teacher=True)

    def test_fit_predict_and_json_content_addressing(self):
        artifact = train_model(self.rows, self.values, feature_manifest=MANIFEST)
        predictions = predict(artifact, self.values, feature_manifest=MANIFEST)
        self.assertLess(predictions[0]["p_relevant"], 0.5)
        self.assertGreater(predictions[-1]["p_relevant"], 0.5)
        self.assertEqual(predictions[0]["mode"], "shadow")
        self.assertEqual(artifact["training"]["report"]["estimate_scope"], "training_fit_not_generalization")
        changed = copy.deepcopy(artifact)
        changed["weights"][0] += 0.01
        with self.assertRaisesRegex(KnowledgeError, "hash mismatch"):
            predict(changed, self.values, feature_manifest=MANIFEST)
        changed_manifest = copy.deepcopy(MANIFEST)
        changed_manifest["encoder_space_id"] = "other-e5-space"
        with self.assertRaisesRegex(KnowledgeError, "manifest do not match"):
            predict(artifact, self.values, feature_manifest=changed_manifest)

    def test_train_only_scaling_does_not_fit_validation_outliers(self):
        validation = [pair("v1", 0), pair("v2", 4)]
        artifact = train_model(self.rows, self.values, feature_manifest=MANIFEST,
                               validation_rows=validation, validation_vectors=np.array([[1000, 1000], [2000, 2000]]))
        self.assertEqual(artifact["mean"], [0.0, 1.0])
        self.assertEqual(artifact["scale"][1], 1.0)
        self.assertTrue(np.isfinite([p["p_relevant"] for p in predict(artifact, np.array([[-1e6, 1e6], [1e6, -1e6]]),
                                                                     feature_manifest=MANIFEST)]).all())

    def test_teacher_training_is_explicit_and_one_class_fails(self):
        self.rows[0]["label"]["origin"] = "teacher"
        with self.assertRaisesRegex(KnowledgeError, "allow_teacher"):
            train_model(self.rows, self.values, feature_manifest=MANIFEST)
        artifact = train_model(self.rows, self.values, feature_manifest=MANIFEST, allow_teacher=True)
        self.assertEqual(artifact["training"]["label_origins"]["teacher"], 1)
        for row in self.rows:
            row["label"]["grade"] = 4
        with self.assertRaisesRegex(KnowledgeError, "both relevant"):
            train_model(self.rows, self.values, feature_manifest=MANIFEST, allow_teacher=True)

    def test_malformed_numeric_artifact_and_nonfinite_features_fail(self):
        artifact = train_model(self.rows, self.values, feature_manifest=MANIFEST)
        with self.assertRaises(KnowledgeError):
            predict(artifact, [[float("nan"), 1]], feature_manifest=MANIFEST)
        with self.assertRaises(KnowledgeError):
            predict(artifact, [[1]], feature_manifest=MANIFEST)
        artifact["scale"][0] = 0
        artifact["model_version"] = "relevance_logreg_" + digest({k: v for k, v in artifact.items() if k != "model_version"})
        with self.assertRaisesRegex(KnowledgeError, "positive"):
            validate_artifact(artifact, feature_manifest=MANIFEST)


if __name__ == "__main__":
    unittest.main()
