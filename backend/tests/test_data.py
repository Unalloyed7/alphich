"""Business invariants for the company context preparation pipeline."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from alphich.data import DataError, PROFILE_FILE, load_companies, prepare_data, sha256_file

FIXTURES = Path(__file__).resolve().parents[2] / "data" / "synthetic"


class PrepareDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.output = Path(cls.temp.name) / "snapshot"
        cls.report = prepare_data(FIXTURES, cls.output, top_n=1)
        cls.contexts = {item["company_id"]: item for item in cls.read_jsonl("contexts.jsonl")}

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    @classmethod
    def read_jsonl(cls, name):
        return [json.loads(line) for line in (cls.output / name).read_text().splitlines()]

    def feature(self, company_id, name):
        return next(item for item in self.contexts[company_id]["features"] if item["name"] == name)

    def test_only_unambiguous_current_clients_above_strict_raw_threshold(self):
        self.assertEqual(set(self.contexts), {"S_BUILD", "S_RENT", "S_SOFTWARE"})
        profiles = {item["id"] for item in self.read_jsonl("companies.jsonl")}
        self.assertNotIn("S_AMBIGUOUS", profiles)
        self.assertIn("S_BOUNDARY", profiles)
        self.assertEqual(self.report["profiles"]["usable_profiles"], 6)
        self.assertEqual(self.report["profiles"]["conflicting_company_ids"], ["S_AMBIGUOUS"])

    def test_self_payments_and_identical_repeats_do_not_inflate_counterparty_metrics(self):
        flow = self.feature("S_BUILD", "payment_incoming")["value"]
        self.assertEqual(flow["amount_raw"], "100")
        self.assertEqual(flow["payment_count"], 3)
        self.assertEqual(flow["counterparty_count"], 2)
        self.assertEqual(flow["normalized_duplicate_rows_skipped"], 1)
        self.assertAlmostEqual(flow["top_counterparty_share"], 0.6)
        self.assertEqual(self.report["payments"]["self_payment_rows"], 1)
        self.assertEqual(self.report["payments"]["additional_directed_pair_rows"], 2)
        self.assertEqual(self.report["payments"]["normalized_duplicate_rows_skipped"], 1)

    def test_top_n_does_not_change_denominator_or_hide_unknown_flows(self):
        flow = self.feature("S_BUILD", "payment_outgoing")["value"]
        self.assertEqual(flow["amount_raw"], "120")
        self.assertEqual(flow["shares_denominator_raw"], "120")
        self.assertEqual(flow["payment_count"], 6)
        self.assertEqual(len(flow["top_counterparties"]), 1)
        self.assertEqual(flow["top_counterparties"][0]["company_id"], "S_UNKNOWN")
        self.assertIsNone(flow["top_counterparties"][0]["industry"])
        self.assertFalse(flow["top_counterparties"][0]["profile_known"])
        self.assertAlmostEqual(flow["known_profile_amount_share"], 50 / 120)
        self.assertAlmostEqual(flow["known_sectors"][0]["share_of_all_flow"], 50 / 120)
        self.assertAlmostEqual(flow["within_known_group_amount_share"], 50 / 120)
        self.assertEqual(flow["additional_pair_rows_aggregated"], 1)
        self.assertIn("repeated_payment_rows_demo_policy", flow["limitations"])

    def test_missing_payments_are_unknown_exposure_not_zero_risk(self):
        flow = self.feature("S_RENT", "payment_incoming")["value"]
        self.assertEqual(flow["counterparty_count"], 0)
        self.assertIsNone(flow["known_profile_amount_share"])
        self.assertIsNone(flow["top_counterparty_share"])
        self.assertIn("no_observed_payments", flow["limitations"])
        self.assertIsNone(flow["currency"])
        self.assertIsNone(flow["period"])

    def test_bad_money_and_fractional_payment_counts_are_quarantined(self):
        self.assertEqual(self.report["payments"]["invalid_rows"], 3)
        errors = self.read_jsonl("quarantine.jsonl")
        self.assertTrue(any("finite nonnegative" in item["reason"] for item in errors))
        self.assertTrue(any("expected an integer" in item["reason"] for item in errors))
        self.assertEqual(self.feature("S_BUILD", "payment_incoming")["value"]["amount_raw"], "100")

    def test_physical_links_do_not_infer_ownership_or_double_count(self):
        links = self.feature("S_BUILD", "person_links")["value"]
        self.assertEqual(links, {"linked_person_count": 2, "other_linked_company_count": 2})
        self.assertEqual(self.report["person_links"]["duplicate_links_skipped"], 1)
        self.assertEqual(self.feature("S_BUILD", "group_profile_count")["value"], 2)
        self.assertIn("person_link_role_unknown", self.contexts["S_BUILD"]["limitations"])

    def test_incomplete_and_conflicting_fields_remain_visible(self):
        companies, report, _ = load_companies(FIXTURES / PROFILE_FILE)
        self.assertIsNone(companies["S_MISSING"].profile["products"]["loan"])
        self.assertIn("missing:REVENUE_COMPANY", companies["S_MISSING"].profile["quality_flags"])
        self.assertIn("fts_country_without_flag", companies["S_BOUNDARY"].profile["quality_flags"])
        self.assertEqual(companies["S_BUILD"].profile["fts"]["countries"], ["КИТАЙ", "ТУРЦИЯ"])
        self.assertEqual(report["rows"], 8)

    def test_artifacts_are_reproducible_and_feature_ids_bind_to_source_version(self):
        other = Path(self.temp.name) / "second"
        prepare_data(FIXTURES, other, top_n=1)
        self.assertEqual((self.output / "manifest.json").read_bytes(), (other / "manifest.json").read_bytes())
        manifest = json.loads((self.output / "manifest.json").read_text())
        for name, digest in manifest["files"].items():
            self.assertEqual(sha256_file(self.output / name), digest)
        incoming = self.feature("S_BUILD", "payment_incoming")
        self.assertEqual(incoming["version"], self.contexts["S_BUILD"]["context_version"])
        self.assertEqual(len(incoming["source_refs"]), 2)

    def test_unknown_or_quarantined_company_does_not_publish_snapshot(self):
        output = Path(self.temp.name) / "invalid-selection"
        with self.assertRaisesRegex(DataError, "Unknown or quarantined"):
            prepare_data(FIXTURES, output, company_ids=["S_AMBIGUOUS"])
        self.assertFalse(output.exists())

    def test_changed_source_creates_new_context_and_evidence_versions(self):
        source = Path(self.temp.name) / "changed-source"
        source.mkdir()
        for path in FIXTURES.iterdir():
            (source / path.name).write_bytes(path.read_bytes())
        with (source / PROFILE_FILE).open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file, delimiter=";")
            fields, rows = reader.fieldnames, list(reader)
        for row in rows:
            if row["SYNTHETIC_U"] == "S_BUILD":
                row["CL_PRODUCT_LOA_MARK"] = "0.0"
        with (source / PROFILE_FILE).open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fields, delimiter=";")
            writer.writeheader()
            writer.writerows(rows)
        output = Path(self.temp.name) / "changed-output"
        prepare_data(source, output, top_n=1)
        changed = next(item for item in (json.loads(line) for line in (output / "contexts.jsonl").read_text().splitlines())
                       if item["company_id"] == "S_BUILD")
        previous = self.contexts["S_BUILD"]
        self.assertNotEqual(changed["profile_version"], previous["profile_version"])
        self.assertNotEqual(changed["context_version"], previous["context_version"])
        self.assertTrue(set(item["id"] for item in changed["features"]).isdisjoint(item["id"] for item in previous["features"]))

    def test_existing_output_is_not_overwritten(self):
        original = (self.output / "manifest.json").read_bytes()
        with self.assertRaisesRegex(DataError, "already exists"):
            prepare_data(FIXTURES, self.output)
        self.assertEqual((self.output / "manifest.json").read_bytes(), original)

    def test_schema_failure_does_not_publish_partial_snapshot(self):
        source = Path(self.temp.name) / "bad-source"
        source.mkdir()
        for path in FIXTURES.iterdir():
            (source / path.name).write_bytes(path.read_bytes())
        (source / PROFILE_FILE).write_text("SYNTHETIC_U;CL_TYPE\nX;CLIENT\n")
        output = Path(self.temp.name) / "bad-output"
        with self.assertRaisesRegex(DataError, "missing columns"):
            prepare_data(source, output)
        self.assertFalse(output.exists())

    def test_explicit_selection_can_include_company_with_unknown_revenue(self):
        output = Path(self.temp.name) / "explicit"
        report = prepare_data(FIXTURES, output, company_ids=["S_MISSING"])
        self.assertEqual(report["selected_company_ids"], ["S_MISSING"])
        context = json.loads((output / "contexts.jsonl").read_text())
        self.assertIsNone(context["profile"]["revenue_raw"])
        self.assertIsNone(context["profile"]["revenue_unit"])


if __name__ == "__main__":
    unittest.main()
