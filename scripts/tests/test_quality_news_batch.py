"""Offline tests: python3 -m unittest discover -s scripts/tests -p test_quality_news_batch.py -v"""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quality_news_batch as q


class QualityNewsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()
        self.sources = [
            {"event_id": "event_1", "url": "https://www.cbr.ru/press/a", "title": "Title 1",
             "grades": {"COMP": 5}, "rationale": {"COMP": "HIDDEN_TEACHER"}},
            {"event_id": "event_2", "url": "https://www.trendforce.com/presscenter/b", "title": "Title 2",
             "grades": {"COMP": 0}},
        ]
        self.save()
        self.private = self.root / "private"
        self.text = "Verified extracted article content " * 5

    def save(self):
        for version in ("v2", "v3"):
            obj = {"version": version, "sources": self.sources, "labels": [
                {"event_id": s["event_id"], "company_id": "COMP", "grade": 5,
                 "rationale": "LEAKED_TEACHER"} for s in self.sources]}
            (self.fixtures / f"real_news_selection_labels.{version}.json").write_text(json.dumps(obj))

    def get(self, url, hosts):
        return SimpleNamespace(url=url, status=200, headers={"content-type": "text/html"}, body=b"<html></html>")

    def extract(self, body, url, kind):
        return self.text

    def run_fetch(self, **kw):
        return q.fetch_batch(fixtures=self.fixtures, private_root=self.private, dataset="v3",
                             batch="batch1", limit=2, delay=.5, get=kw.get("get", self.get),
                             extract=self.extract, get_date=lambda raw: None, sleep=lambda s: None)

    def test_fixture_strip_teacher_data(self):
        src, meta = q.load_fixtures(self.fixtures, "v3")
        self.assertEqual(len(src), 2)
        self.assertNotIn("grade", json.dumps(src))
        self.assertNotIn("HIDDEN_TEACHER", json.dumps(src))
        self.assertEqual(meta["source_count"], 2)

    def test_host_pinning_and_credentials(self):
        self.sources[0]["url"] = "https://localhost/private"
        self.save()
        with self.assertRaisesRegex(ValueError, "pinned"):
            q.load_fixtures(self.fixtures, "v3")
        self.sources[0]["url"] = "https://user:pw@www.cbr.ru/secret"
        self.save()
        with self.assertRaisesRegex(ValueError, "pinned"):
            q.load_fixtures(self.fixtures, "v3")

    def test_draft_no_teacher_no_approval(self):
        result = self.run_fetch()
        self.assertEqual(result["downloaded"], 2)
        draft = (self.private / "batch1" / "draft_articles.json").read_text()
        manifest = (self.private / "batch1" / "manifest.json").read_text()
        self.assertEqual(len(json.loads(draft)), 2)
        self.assertNotIn("LEAKED_TEACHER", draft + manifest)
        self.assertFalse((self.private / "batch1" / "review_articles_APPROVED.json").exists())
        self.assertIsNone(json.loads(draft)["event_1"]["published_at"])

    def test_approve_one_only_after_explicit_choice(self):
        self.run_fetch()
        result = q.approve(batch_dir=self.private / "batch1", event_ids=["event_1"])
        self.assertEqual(result["approved_articles"], 1)
        approved = json.loads(Path(result["output"]).read_text())
        self.assertEqual(list(approved), ["event_1"])
        with self.assertRaises(FileExistsError):
            q.approve(batch_dir=self.private / "batch1", event_ids=["event_2"])

    def test_approve_detects_content_tampering(self):
        self.run_fetch()
        path = self.private / "batch1" / "draft_articles.json"
        val = json.loads(path.read_text())
        val["event_1"]["text"] += " tampered"
        path.write_text(json.dumps(val))
        with self.assertRaisesRegex(ValueError, "changed"):
            q.approve(batch_dir=self.private / "batch1", event_ids=["event_1"])

    def test_fetch_failures_are_explicit_and_no_fake_article(self):
        def broken(url, hosts):
            if "trendforce" in url:
                raise ValueError("request failure details secret")
            return self.get(url, hosts)
        result = self.run_fetch(get=broken)
        self.assertEqual(result["downloaded"], 1)
        self.assertEqual(result["failed"], 1)
        manifest = (self.private / "batch1" / "manifest.json").read_text()
        self.assertIn("fetch_or_extract_failed", manifest)
        self.assertNotIn("secret", manifest)

    def test_batch_never_overwrites(self):
        self.run_fetch()
        with self.assertRaises(FileExistsError):
            self.run_fetch()

    def test_no_fabricated_date(self):
        self.assertTrue(q.article_date_valid(None))
        self.assertTrue(q.article_date_valid("2026-10-08"))
        self.assertFalse(q.article_date_valid("yesterday"))

    def test_preview_escapes_downloaded_html(self):
        preview = q.preview_html({"event_1": {"text": "<script>alert(1)</script>"}}, [{
            "state": "draft", "event_id": "event_1", "title": "Title", "published_at": None,
            "original_url": "https://www.cbr.ru/press/a"}])
        self.assertNotIn("<script>alert", preview)
        self.assertIn("&lt;script&gt;", preview)

    def test_bad_redirect_host(self):
        def hijack(url, hosts):
            return SimpleNamespace(url="https://www.cbr.ru.evil.com/page", status=200,
                                   headers={"content-type": "text/html"}, body=b"text")
        with self.assertRaisesRegex(ValueError, "unsafe_final_url"):
            q.capture_one(self.sources[0], get=hijack, extract=self.extract, get_date=lambda raw: None)


if __name__ == "__main__":
    unittest.main()
