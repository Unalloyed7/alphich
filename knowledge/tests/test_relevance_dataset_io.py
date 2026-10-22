"""Streaming, strict JSON and create-only private dataset checks."""

import json
import tempfile
import unittest
from pathlib import Path

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.dataset import read_dataset, validate_pair, write_dataset


def sample(name="one"):
    return validate_pair({"company": {"company_id": "company", "profile_version": "v1", "facts": [
        {"id": "industry", "name": "okved_code", "value": "62.01", "source_ref": "synthetic fixture"}]},
        "article": {"article_id": name, "source_version": "1", "title": "Source title " + name,
                    "text": "Source full text " + name, "url": "https://example.org/" + name,
                    "published_at": "2026-10-20T00:00:00Z"}, "as_of": "2026-10-22T00:00:00Z", "role": "cfo"})


class DatasetStreamingTests(unittest.TestCase):
    def test_generator_jsonl_roundtrip_prefix_and_private_create_only(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.jsonl"
            summary = write_dataset(path, (sample(str(i)) for i in range(3)), purpose="separate report")
            self.assertEqual(summary["rows"], 3)
            self.assertFalse(summary["metadata_included"])
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(read_dataset(path, limit=1), [sample("0")])
            with path.open("ab") as out:
                out.write(b"invalid unread suffix\n")
            self.assertEqual(len(read_dataset(path, limit=3)), 3)
            with self.assertRaises(KnowledgeError):
                read_dataset(path)
            with self.assertRaises(FileExistsError):
                write_dataset(path, [sample()])

    def test_invalid_or_duplicate_generator_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.jsonl"
            with self.assertRaisesRegex(KnowledgeError, "Duplicate"):
                write_dataset(path, iter([sample(), sample()]))
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_duplicate_json_keys_nonfinite_and_overlong_lines_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.jsonl"
            valid = json.dumps(sample(), ensure_ascii=False)
            for content in (valid[:-1] + ',"role":"ceo"}', valid[:-1] + ',"extra":NaN}'):
                path.write_text(content + "\n", encoding="utf-8")
                with self.assertRaises(KnowledgeError):
                    read_dataset(path)
            path.write_text(valid + "\n", encoding="utf-8")
            with self.assertRaisesRegex(KnowledgeError, "line byte limit"):
                read_dataset(path, max_line_bytes=32)
            with self.assertRaisesRegex(KnowledgeError, "byte limit"):
                read_dataset(path, max_bytes=32)

    def test_envelope_metadata_is_preserved_and_writer_byte_limit_is_atomic(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "pairs.json"
            summary = write_dataset(path, (sample(str(i)) for i in range(2)), purpose="fixture")
            self.assertTrue(summary["metadata_included"])
            self.assertEqual(json.loads(path.read_text())["metadata"], {"purpose": "fixture"})
            self.assertEqual(len(read_dataset(path)), 2)
            fail = Path(folder) / "too-large.jsonl"
            with self.assertRaisesRegex(KnowledgeError, "byte limit"):
                write_dataset(fail, [sample()], max_bytes=32)
            self.assertFalse(fail.exists())


if __name__ == "__main__":
    unittest.main()
