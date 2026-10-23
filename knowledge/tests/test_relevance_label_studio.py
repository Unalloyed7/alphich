"""Human evidence-bound Label Studio annotations become Qwen SFT, without API calls."""
import copy
import json
import tempfile
import unittest
from pathlib import Path

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.dataset import validate_pair, write_dataset
from alphich_kb.relevance_ml.teacher import RESPONSE_SCHEMA, judge_pair
from test_relevance_teacher import judgement, pair


CONFIG_PATH = Path(__file__).resolve().parents[1] / "label-studio" / "relevance-qwen.xml"


def _jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _annotation(response=None, *, annotation_id=11, reviewer=101, cancelled=False):
    response = judgement() if response is None else copy.deepcopy(response)
    result = []
    for field, value in response.items():
        schema = RESPONSE_SCHEMA["properties"][field]
        if field == "grade" or "enum" in schema:
            kind, output = "choices", {"choices": [str(value)]}
        else:
            text = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else value
            kind, output = "textarea", {"text": [] if value is None else [text]}
        result.append({"from_name": field, "to_name": "news_text", "type": kind, "value": output})
    result.append({"from_name": "review_status", "to_name": "news_text", "type": "choices",
                   "value": {"choices": ["accept"]}})
    return {"id": annotation_id, "completed_by": reviewer, "was_cancelled": cancelled, "result": result}


class LabelStudioQwenTests(unittest.TestCase):
    def test_legacy_qwen_messages_are_stable_and_enriched_sft_matches_new_prompt_contract(self):
        from alphich_kb.common import digest
        from alphich_kb.relevance_ml.label_studio import qwen_example, qwen_prompt
        from alphich_kb.relevance_ml.teacher import prepare_judgement
        self.assertEqual(digest(qwen_prompt(pair())[0]), "169ef1ceb7dd286e5e8aa7e0e4bbef721a60ff04e9da3cdec401165cf46c0277")
        row = pair(); row.pop("pair_id")
        row["company"]["facts"].append({"id": "stage", "name": "business_stage_label", "value": "Оптимизация", "source_ref": "fixture"})
        prompt, _ = qwen_prompt(row)
        self.assertEqual(prompt[0]["content"], prepare_judgement(row)["system"])
        example, _ = qwen_example(row, judgement())
        self.assertEqual(example["prompt"], prompt)

    def setUp(self):
        from alphich_kb.relevance_ml.label_studio import export_tasks, import_annotations
        self.export_tasks = export_tasks
        self.import_annotations = import_annotations
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def _export(self, row=None, *, split="train", with_predictions=False, name="bundle"):
        source = self.root / (name + "-source.jsonl")
        write_dataset(source, [pair() if row is None else row])
        bundle = self.root / name
        self.export_tasks(source, bundle, split=split, with_predictions=with_predictions,
                          config_path=str(CONFIG_PATH))
        return bundle

    def _import(self, bundle, annotations=None, *, name="reviewed", tasks=None):
        tasks = copy.deepcopy(json.loads((bundle / "tasks.json").read_text(encoding="utf-8"))) if tasks is None else tasks
        if annotations is not None:
            tasks[0]["annotations"] = annotations
        exported = self.root / (name + "-annotations.json")
        exported.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
        output = self.root / name
        self.import_annotations(bundle, [exported], output, reviewer_namespace="offline-test")
        return output

    def test_manual_review_exports_no_model_answer_and_creates_qwen_sft(self):
        bundle = self._export()
        tasks = json.loads((bundle / "tasks.json").read_text(encoding="utf-8"))
        self.assertEqual(len(tasks), 1)
        self.assertFalse(tasks[0].get("predictions"))
        self.assertTrue((bundle / "pairs.jsonl").is_file())
        self.assertTrue((bundle / "label-config.xml").is_file())
        self.assertTrue((bundle / "manifest.json").is_file())

        output = self._import(bundle, [_annotation()])
        rows = _jsonl(output / "train.jsonl")
        self.assertEqual(len(rows), 1)
        self.assertIsInstance(rows[0]["prompt"], list)
        self.assertEqual(rows[0]["completion"][0]["role"], "assistant")
        answer = json.loads(rows[0]["completion"][0]["content"])
        self.assertEqual(answer["grade"], 5)
        self.assertEqual(answer["company_fact_ids"], ["f1"])
        self.assertEqual(answer["reason"], judgement()["reason"])

        prompt = json.dumps(rows[0]["prompt"], ensure_ascii=False)
        self.assertIn(pair()["article"]["text"], prompt)
        self.assertIn("Грузоперевозки", prompt)
        for private_token in ("EXAMPLE", "article-1", "industry-ref", "region-ref",
                              "source_ref", "profile_version", "input_hash", "pair_id"):
            self.assertNotIn(private_token, prompt)
        self.assertNotIn(judgement()["reason"], prompt)
        self.assertEqual(len(_jsonl(output / "reviewed-pairs.jsonl")), 1)
        self.assertEqual(len(_jsonl(output / "provenance.jsonl")), 1)
        self.assertTrue((output / "manifest.json").is_file())

    def test_teacher_predictions_are_optional_and_never_copied_into_training_prompt(self):
        response = judgement()
        response["reason"] = "SECRET_TEACHER_ANSWER_NOT_A_SOURCE"

        class OfflineTeacher:
            def generate(self, *args):
                return response

        labeled = judge_pair(pair(), OfflineTeacher(), batch_id="offline-test", model_id="fixture-only")
        manual = self._export(labeled, name="manual")
        manual_tasks = json.loads((manual / "tasks.json").read_text(encoding="utf-8"))
        self.assertNotIn(response["reason"], json.dumps(manual_tasks, ensure_ascii=False))
        assisted = self._export(labeled, with_predictions=True, name="assisted")
        assisted_tasks = json.loads((assisted / "tasks.json").read_text(encoding="utf-8"))
        self.assertEqual(len(assisted_tasks[0]["predictions"]), 1)
        self.assertIn(response["reason"], json.dumps(assisted_tasks[0]["predictions"], ensure_ascii=False))

        output = self._import(assisted, [_annotation()])
        trained = _jsonl(output / "train.jsonl")[0]
        self.assertNotIn(response["reason"], json.dumps(trained, ensure_ascii=False))
        self.assertEqual(json.loads(trained["completion"][0]["content"])["reason"], judgement()["reason"])

    def test_prediction_without_human_annotation_is_not_a_training_target(self):
        class OfflineTeacher:
            def generate(self, *args):
                return judgement()

        labeled = judge_pair(pair(), OfflineTeacher(), batch_id="offline-test", model_id="fixture-only")
        bundle = self._export(labeled, with_predictions=True)
        output = self._import(bundle, [])
        self.assertEqual(_jsonl(output / "train.jsonl"), [])
        self.assertEqual(_jsonl(output / "reviewed-pairs.jsonl"), [])

    def test_foreign_quote_is_rejected_without_creating_sft_target(self):
        response = judgement()
        response["news_quotes"] = ["This evidence was not published in the original article."]
        output = self._import(self._export(), [_annotation(response)])
        self.assertEqual(_jsonl(output / "train.jsonl"), [])
        self.assertEqual(_jsonl(output / "reviewed-pairs.jsonl"), [])
        self.assertTrue(json.loads((output / "rejected.json").read_text(encoding="utf-8")))

    def test_edited_task_data_cannot_replace_the_immutable_news_snapshot(self):
        bundle = self._export()
        tasks = json.loads((bundle / "tasks.json").read_text(encoding="utf-8"))
        tasks[0]["data"]["news_text"] = "An attacker substituted a different article."
        tasks[0]["annotations"] = [_annotation()]
        output = self._import(bundle, tasks=tasks)
        self.assertEqual(_jsonl(output / "train.jsonl"), [])
        self.assertTrue(json.loads((output / "rejected.json").read_text(encoding="utf-8")))

    def test_cancelled_annotation_is_not_trained(self):
        output = self._import(self._export(), [_annotation(cancelled=True)])
        self.assertEqual(_jsonl(output / "train.jsonl"), [])

    def test_conflicting_grades_or_explanations_require_adjudication(self):
        for index, changed_field in enumerate(("grade", "reason")):
            with self.subTest(changed_field=changed_field):
                response = judgement()
                response[changed_field] = 4 if changed_field == "grade" else "Иное обоснование применимости события."
                bundle = self._export(name="conflict-" + str(index))
                output = self._import(bundle, [_annotation(), _annotation(response, annotation_id=12, reviewer=102)],
                                      name="conflict-result-" + str(index))
                self.assertEqual(_jsonl(output / "train.jsonl"), [])
                self.assertTrue(json.loads((output / "conflicts.json").read_text(encoding="utf-8")))

    def test_repeated_export_of_the_same_review_does_not_duplicate_training(self):
        bundle = self._export()
        tasks = json.loads((bundle / "tasks.json").read_text(encoding="utf-8"))
        tasks[0]["annotations"] = [_annotation()]
        annotation_paths = []
        for index in range(2):
            path = self.root / ("same-review-" + str(index) + ".json")
            path.write_text(json.dumps(tasks, ensure_ascii=False), encoding="utf-8")
            annotation_paths.append(path)
        output = self.root / "deduplicated"
        self.import_annotations(bundle, annotation_paths, output)
        self.assertEqual(len(_jsonl(output / "train.jsonl")), 1)
        self.assertEqual(len(_jsonl(output / "reviewed-pairs.jsonl")), 1)

    def test_validation_and_test_are_never_written_as_training_examples(self):
        for split in ("validation", "test"):
            with self.subTest(split=split):
                bundle = self._export(split=split, name=split)
                output = self._import(bundle, [_annotation()], name=split + "-result")
                self.assertFalse((output / "train.jsonl").exists())
                self.assertEqual(len(_jsonl(output / (split + ".jsonl"))), 1)

    def test_benchmark_bundle_cannot_show_model_predictions(self):
        for split in ("validation", "test"):
            with self.subTest(split=split), self.assertRaises(KnowledgeError):
                self._export(split=split, with_predictions=True, name=split)

    def test_task_metadata_cannot_move_a_validation_example_into_train(self):
        bundle = self._export(split="validation")
        tasks = json.loads((bundle / "tasks.json").read_text(encoding="utf-8"))
        tasks[0]["data"]["split"] = "train"
        tasks[0]["annotations"] = [_annotation()]
        output = self._import(bundle, tasks=tasks)
        self.assertFalse((output / "train.jsonl").exists())
        self.assertEqual(_jsonl(output / "validation.jsonl"), [])

    def test_export_and_import_refuse_to_overwrite_existing_output(self):
        bundle = self._export()
        source = self.root / "bundle-source.jsonl"
        with self.assertRaises((FileExistsError, KnowledgeError)):
            self.export_tasks(source, bundle, config_path=str(CONFIG_PATH))
        output = self._import(bundle, [_annotation()])
        original = (output / "train.jsonl").read_bytes()
        with self.assertRaises((FileExistsError, KnowledgeError)):
            self.import_annotations(bundle, [self.root / "reviewed-annotations.json"], output)
        self.assertEqual((output / "train.jsonl").read_bytes(), original)

    def _assembly_batch(self, marker, split, *, two_reviewers=True, row=None):
        if row is None:
            row = pair()
            row.pop("pair_id")
            row["article"].update(article_id="article-" + marker, cluster_id="cluster-" + marker,
                                  title="Документы для перевозки: " + marker)
            row["article"]["text"] += " Различающийся контекст публикации: " + marker + "."
            row = validate_pair(row)
        bundle = self._export(row, split=split, name="assembly-" + marker)
        annotations = [_annotation()]
        if two_reviewers:
            annotations.append(_annotation(annotation_id=12, reviewer=102))
        return self._import(bundle, annotations, name="assembly-reviewed-" + marker)

    def test_assemble_distinct_articles_into_frozen_qwen_train_validation_and_test(self):
        from alphich_kb.relevance_ml.label_studio import assemble_sft
        batches = [self._assembly_batch(split, split) for split in ("train", "validation", "test")]
        output = self.root / "assembled"
        assemble_sft([batches[0]], [batches[1]], [batches[2]], output)
        for split in ("train", "validation", "test"):
            examples = _jsonl(output / (split + ".jsonl"))
            self.assertEqual(len(examples), 1)
            self.assertEqual(json.loads(examples[0]["completion"][0]["content"])["grade"], 5)
        self.assertTrue((output / "manifest.json").is_file())

    def test_assemble_rejects_article_ids_reprint_clusters_and_exact_copies_across_splits(self):
        from alphich_kb.relevance_ml.label_studio import assemble_sft
        for shared in ("article_id", "cluster_id", "text"):
            with self.subTest(shared=shared):
                original = pair()
                original.pop("pair_id")
                original["article"]["cluster_id"] = "overlap-cluster-" + shared
                original = validate_pair(original)
                other = copy.deepcopy(original)
                other.pop("pair_id")
                other["company"]["company_id"] = "OTHER_COMPANY"
                other["article"].update(article_id="different-article-" + shared,
                                         cluster_id="different-cluster-" + shared)
                other["article"]["text"] += " Иной дополнительный контекст."
                other["article"][shared] = original["article"][shared]
                other = validate_pair(other)
                train = self._assembly_batch("train-overlap-" + shared, "train", row=original)
                validation = self._assembly_batch("val-overlap-" + shared, "validation", row=other)
                test = self._assembly_batch("test-isolated-" + shared, "test")
                output = self.root / ("assembled-overlap-" + shared)
                with self.assertRaises(KnowledgeError):
                    assemble_sft([train], [validation], [test], output)
                self.assertFalse(output.exists())

    def test_assemble_requires_independent_two_reviewer_evaluation_labels(self):
        from alphich_kb.relevance_ml.label_studio import assemble_sft
        train = self._assembly_batch("train-single-review", "train")
        validation = self._assembly_batch("val-single-review", "validation", two_reviewers=False)
        test = self._assembly_batch("test-independent", "test")
        output = self.root / "assembled-single-review"
        with self.assertRaises(KnowledgeError):
            assemble_sft([train], [validation], [test], output)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
