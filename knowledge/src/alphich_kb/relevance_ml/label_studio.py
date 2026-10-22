"""Offline Label Studio -> Qwen supervised fine-tuning data, without model calls.

One task is one immutable company/news pair. Predictions are optional suggestions;
only accepted, complete human annotations become Qwen prompt/completion targets.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path

from ..common import KnowledgeError, artifact_dir, canonical, digest, file_hash
from .dataset import _strict_json, input_hash, iter_dataset, read_dataset, validate_pair, write_dataset
from .teacher import PROMPT_VERSION, RESPONSE_SCHEMA, SYSTEM_PROMPT, validate_judgement

BUNDLE_SCHEMA = "alphich-label-studio-qwen-v1"
SFT_SCHEMA = "alphich-qwen-sft-v1"
DEFAULT_CONFIG = "knowledge/label-studio/relevance-qwen.xml"
TO_NAME = "news_text"
SPLITS = {"train", "validation", "test"}
ENUM_FIELDS = {name for name, spec in RESPONSE_SCHEMA["properties"].items() if "enum" in spec}
ARRAY_FIELDS = {name for name, spec in RESPONSE_SCHEMA["properties"].items() if spec.get("type") == "array"}
NULLABLE_FIELDS = {"required_action", "valid_from", "valid_until"}


def _read_json(path, maximum=128 * 1024 * 1024):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > maximum:
        raise KnowledgeError("Missing, symlinked or oversized Label Studio JSON file")
    with path.open("rb") as stream:
        content = stream.read(maximum + 1)
    if len(content) > maximum:
        raise KnowledgeError("Label Studio JSON exceeds the byte limit")
    return _strict_json(content)


def _write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    path.chmod(0o600)


def _write_lines(path, values):
    with path.open("w", encoding="utf-8") as stream:
        path.chmod(0o600)
        for value in values:
            stream.write(canonical(value) + "\n")


def _manifest(stage, value):
    value = {**value, "outputs": {p.name: file_hash(p) for p in sorted(stage.iterdir()) if p.is_file()}}
    _write_json(stage / "manifest.json", value)


def _task_data(row, split):
    article = row["article"]
    return {
        "pair_id": row["pair_id"], "input_hash": input_hash(row), "split": split,
        "company_profile": json.dumps({"facts": row["company"]["facts"]}, ensure_ascii=False, indent=2),
        "news_text": article["text"],
        "news_meta": json.dumps({"title": article["title"], "url": article["url"],
                                 "published_at": article["published_at"]}, ensure_ascii=False, indent=2),
        "assessment_context": json.dumps({"as_of": row["as_of"], "role": row["role"],
            "archive_policy": "Архивное событие оценивается условно для данного профиля; это не исторический снимок бизнеса."},
            ensure_ascii=False, indent=2),
    }


def _prediction(judgement, model_version):
    results = []
    for field, value in judgement.items():
        if field == "grade" or field in ENUM_FIELDS:
            kind, payload = "choices", {"choices": [str(value)]}
        else:
            rendered = canonical(value) if field in ARRAY_FIELDS else value
            kind, payload = "textarea", {"text": [] if rendered is None else [rendered]}
        results.append({"from_name": field, "to_name": TO_NAME, "type": kind, "value": payload})
    # review_status is deliberately not prefilled: a person must accept or reject.
    return {"model_version": model_version, "result": results}


def export_tasks(input_path, output_dir, *, split="train", limit=100, offset=0, with_predictions=False,
                 config_path=DEFAULT_CONFIG):
    if split not in SPLITS:
        raise KnowledgeError("Choose train, validation or test before labeling")
    if type(limit) is not int or not 1 <= limit <= 1000:
        raise KnowledgeError("Label Studio task batch limit must be 1..1000")
    if type(offset) is not int or not 0 <= offset <= 1_000_000:
        raise KnowledgeError("Label Studio task offset must be 0..1000000")
    if with_predictions and split != "train":
        raise KnowledgeError("Validation/test tasks must not expose teacher predictions")
    config = Path(config_path)
    if not config.is_file() or config.is_symlink():
        raise KnowledgeError("Missing Label Studio labeling configuration")
    sources = list(islice(iter_dataset(input_path, limit=offset + limit), offset, offset + limit))
    if not sources:
        raise KnowledgeError("No company/news pairs at the requested offset")
    rows, tasks, predictions = [], [], 0
    for source in sources:
        row = copy.deepcopy(source)
        label = row.pop("label", None)
        task = {"data": _task_data(row, split)}
        if with_predictions and label is not None:
            judgement = validate_judgement(label.get("judgement"), row)
            task["predictions"] = [_prediction(judgement, label.get("labeler", "imported-suggestion"))]
            predictions += 1
        rows.append(row)
        tasks.append(task)
    with artifact_dir(output_dir) as stage:
        stage.chmod(0o700)
        write_dataset(stage / "pairs.jsonl", rows)
        _write_json(stage / "tasks.json", tasks)
        (stage / "label-config.xml").write_bytes(config.read_bytes())
        (stage / "label-config.xml").chmod(0o600)
        _manifest(stage, {"schema_version": BUNDLE_SCHEMA, "split": split, "pairs": len(rows),
            "predictions": predictions, "predictions_visible": bool(with_predictions),
            "source_offset": offset,
            "input_hashes": {r["pair_id"]: input_hash(r) for r in rows},
            "target": "qwen_supervised_fine_tuning", "paid_calls": 0})
    return {"tasks": len(rows), "predictions": predictions, "split": split,
            "tasks_file": str(Path(output_dir) / "tasks.json"), "paid_calls": 0}


def _load_bundle(bundle_dir):
    root = Path(bundle_dir)
    manifest = _read_json(root / "manifest.json")
    if (not isinstance(manifest, dict) or manifest.get("schema_version") != BUNDLE_SCHEMA
            or manifest.get("split") not in SPLITS
            or type(manifest.get("predictions_visible")) is not bool
            or set(manifest.get("outputs", {})) != {"tasks.json", "pairs.jsonl", "label-config.xml"}):
        raise KnowledgeError("Unsupported Label Studio Qwen bundle")
    for name, expected in manifest["outputs"].items():
        path = root / name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise KnowledgeError("Label Studio bundle checksum mismatch")
    rows = read_dataset(root / "pairs.jsonl")
    if (manifest.get("pairs") != len(rows)
            or manifest.get("input_hashes") != {r["pair_id"]: input_hash(r) for r in rows}):
        raise KnowledgeError("Label Studio bundle does not match immutable pair inputs")
    if manifest["split"] != "train" and manifest.get("predictions_visible"):
        raise KnowledgeError("Validation/test bundle contains teacher hints")
    return manifest, rows


def _identifier(value, name):
    if type(value) is int and value >= 0:
        return str(value)
    if isinstance(value, str) and 1 <= len(value.strip()) <= 200:
        return value.strip()
    raise KnowledgeError("Missing/invalid " + name)


def _text_result(value):
    texts = value.get("text")
    if not isinstance(texts, list) or len(texts) > 1 or any(not isinstance(t, str) for t in texts):
        raise KnowledgeError("TextArea requires one text submission")
    return texts[0] if texts else ""


def _list_result(text):
    if not text.strip():
        return []
    if text.lstrip().startswith("["):
        value = _strict_json(text)
        if not isinstance(value, list):
            raise KnowledgeError("List field requires a JSON array or one item per line")
        return value
    return [line.strip() for line in text.splitlines() if line.strip()]


def _judgement_from_annotation(annotation, row):
    results = annotation.get("result")
    if not isinstance(results, list) or len(results) > len(RESPONSE_SCHEMA["properties"]) + 1:
        raise KnowledgeError("Annotation requires the full Qwen answer form")
    values = {}
    for result in results:
        if not isinstance(result, dict):
            raise KnowledgeError("Invalid Label Studio result")
        field = result.get("from_name")
        if (field not in set(RESPONSE_SCHEMA["properties"]) | {"review_status"}
                or field in values or result.get("to_name") != TO_NAME):
            raise KnowledgeError("Unknown, duplicate or mismatched annotation control")
        payload = result.get("value")
        if not isinstance(payload, dict):
            raise KnowledgeError("Invalid annotation value")
        if field == "grade" or field in ENUM_FIELDS or field == "review_status":
            choices = payload.get("choices")
            if result.get("type") != "choices" or not isinstance(choices, list) or len(choices) != 1:
                raise KnowledgeError("Each required choice needs exactly one selected value")
            if not isinstance(choices[0], str):
                raise KnowledgeError("Choice value must be a string")
            values[field] = choices[0]
        else:
            if result.get("type") != "textarea":
                raise KnowledgeError("Expected a TextArea control")
            values[field] = _text_result(payload)
    status = values.pop("review_status", None)
    if status == "unusable":
        return None
    if status != "accept":
        raise KnowledgeError("Human acceptance is required before creating an SFT target")
    judgement = {}
    for field in RESPONSE_SCHEMA["properties"]:
        raw = values.get(field)
        if field == "grade":
            if raw not in {str(n) for n in range(6)}:
                raise KnowledgeError("Human grade must be 0..5")
            judgement[field] = int(raw)
        elif field in ENUM_FIELDS:
            judgement[field] = raw
        elif field in ARRAY_FIELDS:
            judgement[field] = _list_result(raw or "")
        elif field in NULLABLE_FIELDS:
            judgement[field] = None if raw is None or raw.strip() in {"", "null"} else raw.strip()
        else:
            judgement[field] = raw
    return validate_judgement(judgement, row)


def qwen_prompt(row):
    """Share the exact semantic input representation between local inference and SFT."""
    row = validate_pair(row)
    aliases = {fact["id"]: "f" + str(i + 1) for i, fact in enumerate(row["company"]["facts"])}
    facts = [{"id": aliases[f["id"]], "name": f["name"], "value": copy.deepcopy(f["value"])}
             for f in row["company"]["facts"]]
    data = {"company": {"facts": facts},
            "article": {key: row["article"][key] for key in ("title", "text", "published_at")},
            "as_of": row["as_of"], "role": row["role"]}
    prompt = [{"role": "system", "content": SYSTEM_PROMPT},
              {"role": "user", "content": canonical({"untrusted_data": data, "response_schema": RESPONSE_SCHEMA})}]
    return prompt, aliases


def qwen_example(row, judgement):
    """Use semantic inputs only; keep opaque IDs and review/teacher metadata outside SFT."""
    row = validate_pair(row)
    judgement = validate_judgement(judgement, row)
    prompt, aliases = qwen_prompt(row)
    target = copy.deepcopy(judgement)
    target["company_fact_ids"] = [aliases[fid] for fid in target["company_fact_ids"]]
    example = {"prompt": prompt,
               "completion": [{"role": "assistant", "content": canonical(target)}]}
    return example, aliases


def import_annotations(bundle_dir, annotation_paths, output_dir, *, reviewer_namespace="label-studio"):
    namespace = _identifier(reviewer_namespace, "reviewer namespace")
    if not annotation_paths:
        raise KnowledgeError("Provide at least one full Label Studio JSON export")
    manifest, originals = _load_bundle(bundle_dir)
    by_id = {row["pair_id"]: row for row in originals}
    candidates = defaultdict(list)
    rejected, conflicts = [], []
    cancelled = unusable = 0
    for path in annotation_paths:
        tasks = _read_json(path)
        if not isinstance(tasks, list) or len(tasks) > 10000:
            raise KnowledgeError("Use a full JSON task export with at most 10000 tasks per file")
        source_hash = file_hash(path)
        for task in tasks:
            pair_id = None
            try:
                if not isinstance(task, dict) or not isinstance(task.get("data"), dict):
                    raise KnowledgeError("Exported task requires original data")
                pair_id = task["data"].get("pair_id")
                if not isinstance(pair_id, str) or pair_id not in by_id:
                    raise KnowledgeError("Exported task is not in this frozen bundle")
                row = by_id[pair_id]
                if task["data"] != _task_data(row, manifest["split"]):
                    raise KnowledgeError("Exported task text/profile/date/hash differs from frozen inputs")
                annotations = task.get("annotations", [])
                if not isinstance(annotations, list) or len(annotations) > 100:
                    raise KnowledgeError("Invalid task annotations")
            except (KnowledgeError, TypeError) as error:
                rejected.append({"pair_id": pair_id if isinstance(pair_id, str) else None,
                                 "error": str(error), "export_sha256": source_hash})
                continue
            for annotation in annotations:
                try:
                    if not isinstance(annotation, dict):
                        raise KnowledgeError("Invalid annotation object")
                    if annotation.get("was_cancelled"):
                        cancelled += 1
                        continue
                    actor = annotation.get("completed_by")
                    if isinstance(actor, dict):
                        actor = actor.get("id")
                    reviewer = namespace + ":" + _identifier(actor, "human annotator ID")
                    annotation_id = _identifier(annotation.get("id"), "annotation ID")
                    judgement = _judgement_from_annotation(annotation, row)
                    if judgement is None:
                        unusable += 1
                        continue
                    candidates[pair_id].append({"reviewer": reviewer, "annotation_id": annotation_id,
                        "judgement": judgement, "export_sha256": source_hash})
                except (KnowledgeError, TypeError, ValueError) as error:
                    rejected.append({"pair_id": pair_id, "error": str(error), "export_sha256": source_hash})
    reviewed, examples, provenance = [], [], []
    for row in originals:
        reviews = candidates.get(row["pair_id"], [])
        if not reviews:
            continue
        answers = {digest(r["judgement"]) for r in reviews}
        if len(answers) != 1:
            conflicts.append({"pair_id": row["pair_id"], "reviews": reviews,
                              "status": "needs_human_resolution_no_training_target"})
            continue
        reviewers = sorted({r["reviewer"] for r in reviews})
        agreed = len(reviewers) >= 2
        judgement = reviews[0]["judgement"]
        updated = copy.deepcopy(row)
        updated["label"] = {"grade": judgement["grade"], "reason": judgement["reason"],
            "origin": "human", "target": "business_relevance", "label_version": SFT_SCHEMA,
            "input_hash": input_hash(row), "judgement": judgement, "reviewers": reviewers,
            "review_status": "agreed" if agreed else "single_review",
            "independently_reviewed": agreed and not manifest["predictions_visible"],
            "timeliness_is_provisional": True,
            "labeled_at": datetime.now(timezone.utc).isoformat()}
        # A teacher-assisted agreement is useful for training, but cannot become blind gold.
        if manifest["predictions_visible"]:
            updated["label"]["review_status"] = "assisted_review"
        example, aliases = qwen_example(updated, judgement)
        reviewed.append(updated)
        examples.append(example)
        provenance.append({"pair_id": row["pair_id"], "input_hash": input_hash(row),
            "split": manifest["split"], "fact_aliases": aliases,
            "prompt_sha256": digest(example["prompt"]), "completion_sha256": digest(example["completion"]),
            "reviews": [{k: r[k] for k in ("reviewer", "annotation_id", "export_sha256")} for r in reviews],
            "teacher_predictions_visible": manifest["predictions_visible"],
            "independent_human_gold": updated["label"]["independently_reviewed"]})
    qwen_filename = manifest["split"] + ".jsonl"
    summary = {"schema_version": SFT_SCHEMA, "split": manifest["split"], "accepted": len(examples),
        "rejected": len(rejected), "conflicts": len(conflicts), "cancelled_annotations": cancelled,
        "unusable_annotations": unusable, "tasks_in_bundle": len(originals),
        "qwen_file": str(Path(output_dir) / qwen_filename), "paid_calls": 0,
        "base_model_candidate": "Qwen/Qwen3.6-35B-A3B", "teacher_prompt_version": PROMPT_VERSION,
        "sft_format": "conversational_prompt_completion; train_on_completion_only",
        "training_ready": bool(examples) and manifest["split"] == "train"}
    with artifact_dir(output_dir) as stage:
        stage.chmod(0o700)
        _write_lines(stage / qwen_filename, examples)
        if reviewed:
            write_dataset(stage / "reviewed-pairs.jsonl", reviewed)
        else:
            _write_lines(stage / "reviewed-pairs.jsonl", [])
        _write_lines(stage / "provenance.jsonl", provenance)
        _write_json(stage / "rejected.json", rejected)
        _write_json(stage / "conflicts.json", conflicts)
        _manifest(stage, {**summary, "source_bundle_manifest_sha256": file_hash(Path(bundle_dir) / "manifest.json")})
    return summary


def assemble_sft(train_batches, validation_batches, test_batches, output_dir, *, company_holdout=False):
    """Combine reviewed batches and reject real cross-split leakage before training."""
    from .evaluation import _independent_human, assert_disjoint_splits

    splits = {}
    inputs = []
    for split, paths in (("train", train_batches), ("validation", validation_batches), ("test", test_batches)):
        if not paths:
            raise KnowledgeError("Qwen dataset assembly requires train, validation and test batches")
        collected = {}
        for folder in paths:
            folder = Path(folder)
            manifest = _read_json(folder / "manifest.json")
            expected_files = {split + ".jsonl", "reviewed-pairs.jsonl", "provenance.jsonl",
                              "rejected.json", "conflicts.json"}
            if (not isinstance(manifest, dict) or manifest.get("schema_version") != SFT_SCHEMA
                    or manifest.get("split") != split or set(manifest.get("outputs", {})) != expected_files
                    or not manifest.get("accepted")):
                raise KnowledgeError("Expected nonempty reviewed Qwen batch for " + split)
            for name, expected in manifest["outputs"].items():
                path = folder / name
                if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
                    raise KnowledgeError("Reviewed Qwen batch checksum mismatch")
            rows = read_dataset(folder / "reviewed-pairs.jsonl")
            examples = []
            with (folder / (split + ".jsonl")).open("rb") as stream:
                while line := stream.readline(1024 * 1024 + 1):
                    if len(line) > 1024 * 1024:
                        raise KnowledgeError("Qwen SFT line exceeds 1 MiB")
                    if line.strip():
                        examples.append(_strict_json(line))
            if len(rows) != manifest["accepted"] or len(examples) != len(rows):
                raise KnowledgeError("Reviewed pair count differs from Qwen targets")
            for row, example in zip(rows, examples):
                label = row.get("label", {})
                if label.get("origin") != "human":
                    raise KnowledgeError("Qwen human SFT assembly requires reviewed human answers")
                if split != "train" and not _independent_human(row):
                    raise KnowledgeError("Validation/test require two independent agreeing human reviewers")
                rebuilt, _ = qwen_example(row, label.get("judgement"))
                if rebuilt != example:
                    raise KnowledgeError("SFT target differs from the reviewed evidence-bound answer")
                previous = collected.get(row["pair_id"])
                if previous is not None and previous[1] != example:
                    raise KnowledgeError("Conflicting reviewed answers for the same training pair")
                collected[row["pair_id"]] = (row, example)
            inputs.append({"split": split, "manifest_sha256": file_hash(folder / "manifest.json")})
        splits[split] = list(collected.values())
    assert_disjoint_splits([r for r, _ in splits["train"]], [r for r, _ in splits["validation"]],
                          [r for r, _ in splits["test"]], company_holdout=company_holdout)
    with artifact_dir(output_dir) as stage:
        stage.chmod(0o700)
        for split, values in splits.items():
            _write_lines(stage / (split + ".jsonl"), [example for _, example in values])
            write_dataset(stage / (split + ".pairs.jsonl"), [row for row, _ in values])
        _manifest(stage, {"schema_version": SFT_SCHEMA, "stage": "assembled_human_sft",
            "counts": {split: len(values) for split, values in splits.items()},
            "company_holdout": company_holdout, "source_batches": inputs,
            "training_model": "Qwen/Qwen3.6-35B-A3B", "paid_calls": 0,
            "evaluation_policy": "held_out_articles_and_declared_blind_two_reviewer_answers",
            "loss_policy": "completion_only; no truncation policy is applied by this exporter"})
    return {"output_dir": str(output_dir), "counts": {s: len(v) for s, v in splits.items()},
            "company_holdout": company_holdout, "paid_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Label Studio human labels -> Qwen SFT, no API calls")
    commands = parser.add_subparsers(dest="command", required=True)
    export = commands.add_parser("export", help="Create immutable Label Studio task bundle")
    export.add_argument("--input", required=True)
    export.add_argument("--output-dir", required=True)
    export.add_argument("--split", choices=sorted(SPLITS), required=True)
    export.add_argument("--limit", type=int, default=100)
    export.add_argument("--offset", type=int, default=0, help="Next batch start within the same frozen input file")
    export.add_argument("--with-predictions", action="store_true", help="Optional prelabels, train split only")
    export.add_argument("--config-path", default=DEFAULT_CONFIG)
    load = commands.add_parser("import", help="Convert accepted complete human answers to Qwen targets")
    load.add_argument("--bundle-dir", required=True)
    load.add_argument("--annotations", nargs="+", required=True, help="Full JSON export(s) from Label Studio")
    load.add_argument("--output-dir", required=True)
    load.add_argument("--reviewer-namespace", default="label-studio")
    assemble = commands.add_parser("assemble", help="Build disjoint train/validation/test for Qwen training")
    assemble.add_argument("--train", nargs="+", required=True, help="Imported train batch directories")
    assemble.add_argument("--validation", nargs="+", required=True)
    assemble.add_argument("--test", nargs="+", required=True)
    assemble.add_argument("--output-dir", required=True)
    assemble.add_argument("--company-holdout", action="store_true", help="Also require disjoint company IDs")
    args = parser.parse_args(argv)
    try:
        if args.command == "export":
            result = export_tasks(args.input, args.output_dir, split=args.split, limit=args.limit,
                                  offset=args.offset, with_predictions=args.with_predictions, config_path=args.config_path)
        elif args.command == "import":
            result = import_annotations(args.bundle_dir, args.annotations, args.output_dir,
                                        reviewer_namespace=args.reviewer_namespace)
        else:
            result = assemble_sft(args.train, args.validation, args.test, args.output_dir,
                                  company_holdout=args.company_holdout)
        print(json.dumps(result, ensure_ascii=False))
        return 0
    except (KnowledgeError, OSError) as error:
        print(json.dumps({"status": "error", "error": str(error)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
