"""A tiny parallel job uses only fake responses and temporary SQLite files."""
import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.collect_job import _groups, collect_job, iter_job_articles, job_summary
from alphich_kb.relevance_ml.corpus import collect
from alphich_kb.relevance_ml import collect_job as job_module
from alphich_kb.relevance_ml.corpus import _database

from test_relevance_corpus import Clock, FakeFetcher, article, rss, source


class CollectJobTests(unittest.TestCase):
    def test_bridge_merges_both_existing_clusters_and_export_uses_frozen_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"schema_version": "alphich-relevance-sources-v1", "sources": [source()]}))

            def insert(database, names):
                with _database(database) as db:
                    for name in names:
                        text = "Текст материала " + name
                        content_hash = hashlib.sha256(text.encode()).hexdigest()
                        db.execute("INSERT INTO articles VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (
                            "article_" + name, "https://news.example.test/news/" + name, "one", "Деловой материал " + name,
                            "деловой материал " + name, text, content_hash, "source_cluster_" + name, "[]",
                            "2026-10-20T10:00:00+00:00", "2026-10-21T10:00:00+00:00", "{}"))

            def seed_collector(_, database, **kwargs):
                insert(database, ["A", "B", "C"])
                return {"requests": 0, "new": 3, "response_bytes": 0, "byte_budget_used": 0, "stop_reason": "corpus_goal"}

            collect_job(manifest, root / "job", goal=3, collector=seed_collector, max_runtime_seconds=10)
            # A and B are below the .82 threshold. C is above it for both:
            # assigning C to only the first match would leak B/C across splits.
            signatures = {"A": list(range(128)), "B": list(range(8)) + list(range(30, 150)),
                          "C": list(range(8)) + list(range(19, 139))}
            original_reader = job_module._frozen_articles

            def append_during_second_pass(database, records):
                insert(database, ["D"])
                yield from original_reader(database, records)

            with patch.object(job_module, "_signature", side_effect=lambda text: [f"{value:016x}" for value in signatures[text[-1]]]), \
                    patch.object(job_module, "_frozen_articles", side_effect=append_during_second_pass):
                exported = list(iter_job_articles(root / "job"))
            self.assertEqual({row["article_id"] for row in exported}, {"article_A", "article_B", "article_C"})
            self.assertEqual(len({row["cluster_id"] for row in exported}), 1)
            self.assertEqual(job_summary(root / "job")["total"], 4)

    def test_resume_rejects_altered_group_manifest_before_scheduling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "sources.json"
            manifest.write_text(json.dumps({"schema_version": "alphich-relevance-sources-v1", "sources": [source()]}))

            def no_network_collector(_, database, **kwargs):
                with _database(database):
                    pass
                return {"requests": 0, "new": 0, "response_bytes": 0, "stop_reason": "queue_exhausted"}

            collect_job(manifest, root / "job", goal=2, collector=no_network_collector, max_runtime_seconds=10)
            group_file = next((root / "job").glob("group_*.sources.json"))
            original = group_file.read_text()
            changed = json.loads(original)
            changed["sources"][0]["min_interval_seconds"] = 0.5
            group_file.write_text(json.dumps(changed))
            with patch.object(job_module, "collect") as network:
                with self.assertRaisesRegex(KnowledgeError, "group manifest changed"):
                    collect_job(manifest, root / "job", goal=2, max_runtime_seconds=10)
                network.assert_not_called()
            group_file.write_text(original)
            frozen_root = root / "job" / "sources.json"
            changed = json.loads(frozen_root.read_text())
            changed["sources"][0]["name"] = "Changed frozen source"
            frozen_root.write_text(json.dumps(changed))
            with self.assertRaisesRegex(KnowledgeError, "source manifest or host groups changed"):
                collect_job(manifest, root / "job", goal=2, collector=no_network_collector, max_runtime_seconds=10)

    def test_overlapping_allowed_hosts_are_one_worker_group(self):
        groups = _groups([{"id": "one", "allowed_hosts": {"a.example", "b.example"}},
                          {"id": "two", "allowed_hosts": {"b.example", "c.example"}},
                          {"id": "three", "allowed_hosts": {"d.example"}}])
        self.assertEqual(len(groups), 2)
        self.assertIn(["one", "two"], [group["source_ids"] for group in groups])

    def test_two_groups_collect_and_resume_without_refetch_or_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "sources.json"
            configs = [source("one.example.test", "one"), source("two.example.test", "two")]
            manifest.write_text(json.dumps({"schema_version": "alphich-relevance-sources-v1", "sources": configs}))
            values = {}
            for config in configs:
                host = config["allowed_hosts"][0]
                values[f"https://{host}/robots.txt"] = (b"User-agent: *\nAllow: /", "text/plain")
                values[f"https://{host}/rss"] = rss(host, ["one"])
                values[f"https://{host}/news/one"] = article("Деловая новость " + host, suffix=host)
            fetcher = FakeFetcher(values)
            clock = Clock()

            def fake_collect(manifest_path, database, **kwargs):
                return collect(manifest_path, database, fetcher=fetcher, clock=clock, **kwargs)

            progress = []
            result = collect_job(manifest, root / "job", goal=2, max_workers=2, max_runtime_seconds=10,
                                 max_articles_per_batch=1, progress=progress.append, collector=fake_collect)
            self.assertEqual(result["total"], 2)
            self.assertEqual(result["labels_created"], 0)
            self.assertEqual(result["stop_reason"], "corpus_goal")
            self.assertEqual(len(list(iter_job_articles(root / "job"))), 2)
            self.assertEqual(job_summary(root / "job")["total"], 2)
            first_calls = list(fetcher.calls)
            again = collect_job(root / "job" / "sources.json", root / "job", goal=2, max_workers=2,
                                max_runtime_seconds=10, collector=fake_collect)
            self.assertEqual(again["requests"], 0)
            self.assertEqual(fetcher.calls, first_calls)
            self.assertTrue(any(row["event"] == "batch_completed" for row in progress))
            changed = json.loads(manifest.read_text())
            changed["sources"][0]["name"] = "Different publisher metadata"
            manifest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(KnowledgeError, "frozen"):
                collect_job(manifest, root / "job", goal=2, collector=fake_collect)


if __name__ == "__main__":
    unittest.main()
