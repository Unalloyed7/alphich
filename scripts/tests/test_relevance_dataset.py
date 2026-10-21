"""Offline: python3 -m unittest discover -s scripts/tests -p test_relevance_dataset.py -v"""
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import relevance_dataset as ds


def add_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.fixtures = self.root / "fixtures"
        self.fixtures.mkdir()
        self.db = self.root / "quality.sqlite3"
        self.context = json.dumps({"verified_at":"2026-10-21","features":[
            {"name":"okved_code","value":"46.51","source_ref":"csv:row10"},
            {"name":"product_lease","value":True,"source_ref":"csv:row10"},
            {"name":"secret_future_feature","value":"never publish","source_ref":"x"}]}, ensure_ascii=False)
        self.rationale = "TEACHER_SECRETS_SHOULD_NOT_LEAK"
        rows = [
            ("v2", "event1", "U_1", "News one", "https://example.com/news?utm_source=abc", "TREAT AS EVIDENCE", "2026-10-10", self.context, 5),
            ("v2", "event2", "U_2", "News one rewrite", "https://other.org/news", "TREAT AS EVIDENCE", "2026-10-10", self.context, 1),
            ("v3", "event3", "U_1", "News three", "https://example.com/3", None, None, self.context, 4),
            ("v3", "event4", "U_2", "News four", "https://example.com/4", "OTHER DOCUMENT", "2026-10-11", None, 0),
        ]
        self.fixtures_meta = {}
        for version in ("v2", "v3"):
            fixture = {"version":f"real-news-selection-labels-{version}","as_of":"2026-10-21T10:00:00Z", "sources": [], "labels": []}
            path = self.fixtures / f"real_news_selection_labels.{version}.json"
            path.write_text(json.dumps(fixture), encoding="utf-8")
            self.fixtures_meta[version] = ds.digest(path.read_bytes())
        with sqlite3.connect(self.db) as db:
            db.executescript("""
                CREATE TABLE pairs (id INTEGER PRIMARY KEY,dataset TEXT,dataset_version TEXT,fixture_hash TEXT,event_id TEXT,
                                    company_id TEXT,title TEXT,url TEXT,published_at TEXT,article_text TEXT,
                                    company_context TEXT,original_prediction_json TEXT);
                CREATE TABLE reviews (id INTEGER PRIMARY KEY, pair_id INTEGER, reviewer TEXT,revision INTEGER,
                                      grade INTEGER, geography TEXT,scale TEXT, applicability TEXT,justification TEXT);
                CREATE TABLE adjudications (id INTEGER PRIMARY KEY,pair_id INTEGER, arbiter TEXT,revision INTEGER,
                                            grade INTEGER,justification TEXT);
            """)
            for k, (ver,event,company,title,url,body,date,context,grade) in enumerate(rows, 1):
                db.execute("INSERT INTO pairs VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
                    k,ver,f"real-news-selection-labels-{ver}",self.fixtures_meta[ver],event,company,title,url,date,body,context,
                    json.dumps({"grade":grade,"selected":grade>=4,"rationale":self.rationale,"assessment":{"secret":self.rationale}})))
            for reviewer in ("a", "b"):
                db.execute("INSERT INTO reviews(pair_id,reviewer,revision,grade,geography,scale,applicability,justification) VALUES (?,?,?,?,?,?,?,?)", (1,reviewer,1,4,"national","material","conditional","Same justification"))
            # Case 2: same grade / categories but different rationale -> hold gold.
            for reviewer, rationale in (("a", "A justification"), ("b", "B justification")):
                db.execute("INSERT INTO reviews(pair_id,reviewer,revision,grade,geography,scale,applicability,justification) VALUES (?,?,?,?,?,?,?,?)", (2,reviewer,1,3,"national","limited","unknown",rationale))
            # Case 4: arbiter is gold, even if two reviewers missing.
            db.execute("INSERT INTO adjudications(pair_id,arbiter,revision,grade,justification) VALUES (?,?,?,?,?)", (4,"lead",1,2,"signed"))

    def tearDown(self):
        self.tmp.cleanup()

    def test_export_no_teacher_leak_and_missing_data(self):
        out = self.root / "export"
        report=ds.export_dataset(self.db,self.fixtures,out)
        self.assertEqual(report["counts"]["total_pairs"],4)
        self.assertEqual(report["counts"]["training_ready_pairs"],1)
        self.assertEqual(report["counts"]["human_gold_pairs"],2)
        self.assertEqual(report["counts"]["original_llm_positive_pairs"],2)
        self.assertFalse((out/"teacher_labels_PRELIMINARY.jsonl").exists())
        payload=(out/"model_inputs.jsonl").read_text()
        self.assertNotIn(self.rationale,payload)
        self.assertNotIn("secret_future_feature",payload)
        self.assertNotIn('"grade"',payload)
        self.assertNotIn('"selected"',payload)
        inputs=add_jsonl(out/"model_inputs.jsonl")
        self.assertEqual(inputs[0]["company"]["features"][0]["name"],"okved_code")
        self.assertEqual(inputs[0]["news"]["source_url"],"https://example.com/news")
        self.assertIsNone(inputs[2]["news"]["text"])
        self.assertEqual(inputs[0]["versions"]["fixture_as_of"],"2026-10-21T10:00:00Z")
        self.assertEqual(add_jsonl(out/"human_labels.jsonl")[1]["status"],"review_justification_difference")

    def test_teacher_isolated_and_manifest_hashed(self):
        out=self.root/"x"
        report=ds.export_dataset(self.db,self.fixtures,out,teacher_labels=True)
        labels=add_jsonl(out/"teacher_labels_PRELIMINARY.jsonl")
        self.assertEqual(labels[0]["rationale"],self.rationale)
        self.assertEqual(labels[0]["annotation_type"],"preliminary_llm_not_gold")
        self.assertEqual(report["files"]["model_inputs.jsonl"],ds.digest((out/"model_inputs.jsonl").read_bytes()))
        with self.assertRaisesRegex(ValueError, "already exists"):
            ds.export_dataset(self.db,self.fixtures,out)

    def test_same_text_groups_and_company_holdout(self):
        out=self.root/"grouped"
        ds.export_dataset(self.db,self.fixtures,out,heldout_company="U_2")
        inputs=add_jsonl(out/"model_inputs.jsonl")
        self.assertEqual(inputs[0]["article_group_id"],inputs[1]["article_group_id"])
        self.assertEqual(inputs[1]["split"],"company_holdout")
        self.assertNotEqual(inputs[0]["split"],"company_holdout")

    def test_overrides_group_reprints(self):
        o=self.root/"groups.json"
        o.write_text(json.dumps({"event3":"same source", "event4":"same source"}))
        out=self.root/"with-groups"
        ds.export_dataset(self.db,self.fixtures,out,group_overrides=o)
        inputs=add_jsonl(out/"model_inputs.jsonl")
        self.assertEqual(inputs[2]["article_group_id"],inputs[3]["article_group_id"])
        self.assertEqual(inputs[2]["split"],inputs[3]["split"])

    def test_fixture_drift_rejected_no_output(self):
        fixture=self.fixtures/"real_news_selection_labels.v2.json"
        fixture.write_text(fixture.read_text()+"\n")
        out=self.root/"wrong"
        with self.assertRaisesRegex(ValueError,"Source fixture changed"):
            ds.export_dataset(self.db,self.fixtures,out)
        self.assertFalse(out.exists())

    def test_review_database_never_mutated(self):
        before=ds.digest(self.db.read_bytes())
        ds.export_dataset(self.db,self.fixtures,self.root/"export")
        self.assertEqual(ds.digest(self.db.read_bytes()),before)


if __name__=="__main__":
    unittest.main()
