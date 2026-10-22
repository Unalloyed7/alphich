import csv
import tempfile
import unittest
from pathlib import Path

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.sampling import load_company_snapshots, okved_sector, sample_pairs


def company(company_id, code):
    return {"company_id": company_id, "profile_version": "test-v1", "facts": [
        {"id": company_id + "_industry", "name": "okved_code", "value": code, "source_ref": "test:row:1"}]}


def article(article_id, sector="construction", text=None, cluster=None):
    return {"article_id": article_id, "source_version": "1", "cluster_id": cluster or article_id,
            "title": "Current event " + article_id, "text": text or "Full article body " + article_id,
            "url": "https://example.test/" + article_id, "published_at": "2026-10-20T00:00:00Z", "topics": [sector]}


class RelevanceSamplingTests(unittest.TestCase):
    def test_csv_quarantines_every_conflicting_version_and_binds_organizer_units(self):
        fields = ["SYNTHETIC_U", "MAINOKVED_CODE", "REGION", "REVENUE_COMPANY", "CL_PRODUCT_LOA_MARK"]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "companies.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
                writer.writeheader()
                writer.writerows([
                    dict(zip(fields, ["good", "41.20", "Москва", "1234567", "1"])),
                    dict(zip(fields, ["good", "41.20", "Москва", "1234567", "1"])),
                    dict(zip(fields, ["bad", "62.01", "Москва", "1", "0"])),
                    dict(zip(fields, ["bad", "62.01", "Алтай", "1", "0"])),
                ])
            companies, census = load_company_snapshots(path)
        self.assertEqual([item["company_id"] for item in companies], ["good"])
        self.assertEqual(census["quarantined_company_count"], 1)
        self.assertEqual(census["quarantined_source_rows"], 2)
        facts = {fact["name"]: fact for fact in companies[0]["facts"]}
        self.assertTrue(facts["loan"]["value"])
        self.assertEqual(facts["revenue_raw"]["value"], "1234567")
        self.assertEqual(facts["revenue_unit"]["value"], "RUB")
        self.assertEqual(facts["revenue_currency"]["value"], "RUB")
        self.assertEqual(facts["revenue_period"]["value"], "2025")
        self.assertEqual(facts["revenue_scope"]["value"], "company_or_group_unknown")
        self.assertNotIn("operating_regions", facts)
        self.assertIn("csv_sha256:", facts["registration_region"]["source_ref"])

    def test_sampling_is_deterministic_bounded_company_conditioned_and_unlabeled(self):
        companies = [company("c1", "41.20"), company("c2", "43.11"), company("r1", "68.20"),
                     company("m1", "25.11"), company("it1", "62.01"), company("t1", "47.11")]
        articles = [article(str(index)) for index in range(10)]
        first, report = sample_pairs(articles, companies, as_of="2026-10-22T00:00:00Z", limit=12)
        second, _ = sample_pairs(list(reversed(articles)), list(reversed(companies)),
                                 as_of="2026-10-22T00:00:00Z", limit=12)
        self.assertEqual([row["pair_id"] for row in first], [row["pair_id"] for row in second])
        self.assertEqual(len(first), 12)
        self.assertTrue(all("label" not in row for row in first))
        self.assertEqual(report["labels_generated"], 0)
        self.assertEqual(report["pair_kinds"]["same_sector"], 4)
        self.assertGreaterEqual(report["sampled_company_count"], 4)
        for offset in range(0, 12, 3):
            self.assertEqual(len({row["company"]["company_id"] for row in first[offset:offset + 3]}), 3)

    def test_reprints_are_not_sampled_twice_and_bad_date_fails_early(self):
        companies = [company("c1", "41.20"), company("it", "62.01")]
        articles = [article("x", text="Shared original text"), article("y", text=" shared ORIGINAL  text ")]
        rows, report = sample_pairs(articles, companies, as_of="2026-10-22T00:00:00Z")
        self.assertEqual(len(rows), 2)
        self.assertEqual(report["reprints_or_duplicates_skipped"], 1)
        with self.assertRaises(KnowledgeError):
            sample_pairs(articles, companies, as_of="2026-10-22")

    def test_okved_sections_keep_unknown_separate(self):
        self.assertEqual(okved_sector("01.11"), "agriculture")
        self.assertEqual(okved_sector("62.01"), "information")
        self.assertEqual(okved_sector("49.41"), "transport")
        self.assertEqual(okved_sector("bad value"), "unknown")


if __name__ == "__main__":
    unittest.main()
