"""Focused tests use no model assets, API calls, network or GPU."""

import copy
import hashlib
import unittest

import numpy as np

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.features import FeatureBuilder


class FakeEncoder:
    dimension = 4
    max_tokens = 100
    space_id = "fake-e5-space-v1"

    def __init__(self):
        self.queries = []

    def count_tokens(self, text):
        return len(text) + 2

    def encode_query(self, text):
        assert self.count_tokens("query: " + text) <= self.max_tokens
        self.queries.append(text)
        values = np.asarray(list(hashlib.sha256(text.encode()).digest()[:4]), dtype=np.float32) + 1
        return values / np.linalg.norm(values)


def pair():
    return {
        "pair_id": "private-pair-id",
        "company": {"company_id": "private-company-id", "profile_version": "1", "facts": [
            {"id": "industry-fact", "name": "industry", "value": {"code": "62", "name": "IT"}, "source_ref": "private-company-source"},
        ]},
        "article": {"article_id": "private-article-id", "source_version": "1", "cluster_id": "private-cluster",
                    "title": "Новая программа", "text": "Текст новости о программном обеспечении.",
                    "url": "https://example.test/one", "published_at": "2026-10-20T12:00:00Z"},
        "as_of": "2026-10-22T12:00:00Z", "role": "cfo",
        "label": {"grade": 4, "origin": "teacher", "reason": "Субъективная причина"},
    }


class RelevanceFeatureTests(unittest.TestCase):
    def test_source_employee_conflicts_and_opaque_labels_are_not_invented_business_signals(self):
        row = pair()
        row["company"]["facts"] += [
            {"id": key, "name": key, "value": value, "source_ref": "synthetic organizer fixture"}
            for key, value in (("employees_fns_band", "101-250"), ("employees_spark_band", "251-500"),
                               ("employees_band", "101-250"), ("company_age_band", "8-15"),
                               ("business_stage_label", "OPAQUE_STAGE_SECRET"),
                               ("negative_mark_raw", "OPAQUE_NEGATIVE_SECRET"), ("client_type", "OPAQUE_CLIENT_SECRET"),
                               ("city_registration", "Демо"))]
        encoder = FakeEncoder(); builder = FeatureBuilder(encoder)
        values = dict(zip(builder.feature_names, builder.transform([row])[0]))
        self.assertEqual(values["employees_fns_band_known"], 1)
        self.assertEqual(values["employees_spark_band_known"], 1)
        self.assertEqual(values["employee_sources_conflict"], 1)
        self.assertEqual(values["employees_band_known"], 0)
        self.assertEqual(values["operating_regions_known"], 0)
        encoded = "".join(encoder.queries)
        for raw in ("OPAQUE_STAGE_SECRET", "OPAQUE_NEGATIVE_SECRET", "OPAQUE_CLIENT_SECRET"):
            self.assertNotIn(raw, encoded)
        self.assertIn("101-250", encoded)
        self.assertIn("251-500", encoded)
        self.assertIn("дата наблюдения неизвестна", encoded)

    def test_complete_article_coverage_and_tail_changes_prediction_inputs(self):
        row = pair()
        row["article"]["text"] = ("abcdefЖЗИЙКЛМНОПРСТ" * 20) + " УНИКАЛЬНЫЙ КОНЕЦ"
        encoder = FakeEncoder()
        builder = FeatureBuilder(encoder)
        first = builder.transform([row])
        # The short sourced company context occupies exactly one query. Every
        # subsequent query represents a contiguous part of the full article.
        self.assertEqual("".join(encoder.queries[1:]), row["article"]["title"] + "\n\n" + row["article"]["text"])
        row["article"]["text"] = row["article"]["text"].replace("КОНЕЦ", "ФИНАЛ")
        second = builder.transform([row])
        self.assertFalse(np.array_equal(first, second))
        self.assertEqual(first.shape, (1, len(builder.feature_names)))

    def test_teacher_labels_reasons_ids_and_unknown_company_fields_are_not_features(self):
        original = pair()
        changed = copy.deepcopy(original)
        changed["label"] = {"grade": 0, "origin": "human", "reason": "Совсем другая причина"}
        changed["teacher_grade"] = 5
        changed["reason"] = "Утечка ответа"
        changed["company"]["company_id"] = "different-company"
        changed["company"]["reason"] = "Ещё одна утечка"
        changed["company"]["facts"][0]["id"] = "different-fact-id"
        changed["company"]["facts"][0]["source_ref"] = "different-source-id"
        changed["article"]["article_id"] = "different-article"
        changed["article"]["url"] = "https://another.example/path"
        changed["company"]["facts"].append({"id": "extra", "name": "teacher_reason", "value": "Признак критичности", "source_ref": "s"})
        builder = FeatureBuilder(FakeEncoder())
        np.testing.assert_array_equal(builder.transform([original]), builder.transform([changed]))

    def test_registration_is_not_operating_territory_and_unbacked_products_are_unknown(self):
        row = pair()
        row["company"]["facts"] += [
            {"id": "registration", "name": "registration_region", "value": "Алтай", "source_ref": "registry"},
            {"id": "loan", "name": "loan", "value": True},
            {"id": "deposit", "name": "deposit", "value": False, "source_ref": "products"},
        ]
        builder = FeatureBuilder(FakeEncoder())
        values = dict(zip(builder.feature_names, builder.transform([row])[0]))
        self.assertEqual(values["registration_region_known"], 1)
        self.assertEqual(values["operating_regions_known"], 0)
        self.assertEqual(values["operating_region_mentioned"], 0)
        self.assertEqual(values["loan_known"], 0)
        self.assertEqual(values["deposit_known"], 1)
        self.assertEqual(values["deposit_positive"], 0)

    def test_conflicting_products_are_unknown_not_last_record_wins(self):
        row = pair()
        row["company"]["facts"] += [
            {"id": "one", "name": "loan", "value": True, "source_ref": "one"},
            {"id": "two", "name": "loan", "value": False, "source_ref": "two"},
        ]
        builder = FeatureBuilder(FakeEncoder())
        values = dict(zip(builder.feature_names, builder.transform([row])[0]))
        self.assertEqual(values["loan_known"], 0)
        self.assertEqual(values["loan_positive"], 0)

    def test_nested_trade_and_products_have_explicit_masks_without_revenue_amounts(self):
        row = pair()
        row["company"]["facts"] += [
            {"id": "trade", "name": "fts", "value": {"flag": True, "countries": ["Китай"]}, "source_ref": "trade-profile"},
            {"id": "products", "name": "products", "value": {"guarantee": True, "rko": False}, "source_ref": "bank-profile"},
            {"id": "raw-revenue", "name": "revenue_raw", "value": 1000, "source_ref": "financials"},
            {"id": "revenue-period", "name": "revenue_period", "value": "2025", "source_ref": "financials"},
        ]
        builder = FeatureBuilder(FakeEncoder())
        features = builder.transform([row])
        values = dict(zip(builder.feature_names, features[0]))
        self.assertEqual(values["fts_positive"], 1)
        self.assertEqual(values["trade_countries_known"], 1)
        self.assertEqual(values["guarantee_positive"], 1)
        self.assertEqual(values["rko_known"], 1)
        self.assertEqual(values["rko_positive"], 0)
        self.assertEqual(values["raw_revenue_present"], 0)
        self.assertEqual(values["revenue_unit_known"], 0)
        self.assertEqual(values["revenue_period_known"], 1)

    def test_business_fit_features_exclude_publication_age_and_use_only_qualified_scale(self):
        row = pair()
        row["company"]["facts"] += [
            {"id": name, "name": name, "value": value, "source_ref": "organizer clarification"}
            for name, value in [("revenue_raw", "25000000000"), ("revenue_unit", "RUB"),
                                ("revenue_currency", "RUB"), ("revenue_period", "2025"),
                                ("revenue_scope", "company_or_group_unknown")]]
        builder = FeatureBuilder(FakeEncoder())
        recent = builder.transform([row])[0]
        row["article"]["published_at"] = "2025-01-01T00:00:00Z"
        np.testing.assert_array_equal(recent, builder.transform([row])[0])
        values = dict(zip(builder.feature_names, recent))
        self.assertNotIn("age_days_log1p", values)
        self.assertEqual(values["reported_revenue_scale_known"], 1)
        self.assertGreater(values["reported_revenue_rub_2025_log1p"], 20)
        self.assertEqual(values["revenue_scope_known"], 0)
        next(f for f in row["company"]["facts"] if f["name"] == "revenue_currency")["value"] = "USD"
        values = dict(zip(builder.feature_names, builder.transform([row])[0]))
        self.assertEqual(values["reported_revenue_scale_known"], 0)
        self.assertEqual(values["reported_revenue_rub_2025_log1p"], 0)

    def test_oversize_and_future_article_rejected_instead_of_silent_truncation(self):
        row = pair()
        with self.assertRaisesRegex(KnowledgeError, "truncation"):
            FeatureBuilder(FakeEncoder(), max_article_chars=10).transform([row])
        row["article"]["published_at"] = "2027-01-01T00:00:00Z"
        with self.assertRaisesRegex(KnowledgeError, "future"):
            FeatureBuilder(FakeEncoder()).transform([row])

    def test_manifest_pins_embedding_space_and_transform_configuration(self):
        builder = FeatureBuilder(FakeEncoder())
        manifest = builder.manifest
        self.assertEqual(manifest["dimension"], len(builder.feature_names))
        self.assertEqual(manifest["configuration"]["article_prefix"], "query: ")
        manifest["feature_names"].clear()
        self.assertTrue(builder.manifest["feature_names"])
        changed = FakeEncoder()
        changed.space_id = "another-space"
        self.assertNotEqual(builder.manifest, FeatureBuilder(changed).manifest)
        self.assertEqual(builder.transform([]).shape, (0, len(builder.feature_names)))


if __name__ == "__main__":
    unittest.main()
