"""Bounded, resumable two-model job; no retries and no independent budget."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ..common import KnowledgeError, digest, file_hash, read_json
from .api_runner import batch_lock
from .automatic import queued_rows, validate_labeling_input
from .dataset import read_dataset, write_private_json


def run_step(arguments):
    # Secrets stay in the inherited server environment, never command arguments.
    completed = subprocess.run([sys.executable, "-m", "alphich_kb.relevance_ml", *arguments],
                               capture_output=True, text=True)
    try:
        value = json.loads(completed.stdout)
    except (ValueError, TypeError):
        raise KnowledgeError("Labeling child returned no bounded JSON result; inspect its private checkpoints") from None
    if not isinstance(value, dict):
        raise KnowledgeError("Invalid labeling child result")
    if completed.returncode and value.get("status") != "failed":
        return {"status": "failed", "error": "labeling_child_nonzero_exit"}
    return value


def run_automatic_job(*, input_path, pilot_plan, queue_path, output_dir, batch_prefix,
                      primary_model, verifier_model, limit=24, prompt_variant="case-calibrated",
                      max_cost_rub="5", max_seconds=900, live=False, shared_ledger=False,
                      train_shadow=False, step=run_step):
    if live and not shared_ledger:
        raise KnowledgeError("Automatic paid labeling requires the existing shared ledger")
    if not primary_model or not verifier_model or primary_model == verifier_model:
        raise KnowledgeError("Automatic review requires two different allowed models")
    if not isinstance(batch_prefix, str) or not 1 <= len(batch_prefix) <= 160:
        raise KnowledgeError("Explicit bounded batch prefix is required")
    validate_labeling_input(pilot_plan, input_path)
    selected = queued_rows(input_path, queue_path, limit=limit)
    if not selected:
        raise KnowledgeError("Automatic labeling queue is empty")
    destination = Path(output_dir)
    identity = {"input_sha256": file_hash(input_path), "plan_sha256": file_hash(pilot_plan),
        "queue_sha256": file_hash(queue_path), "batch_prefix": batch_prefix,
        "primary_model": primary_model, "verifier_model": verifier_model, "limit": limit,
        "prompt_variant": prompt_variant, "max_cost_rub_per_model": str(max_cost_rub),
        "max_seconds_per_model": max_seconds, "train_shadow": train_shadow}
    manifest = {"schema_version": "alphich-automatic-labeling-job-v1", **identity,
                "job_id": "autolabel_" + digest(identity)}
    report = {"job_id": manifest["job_id"], "status": "dry_run" if not live else "running", "steps": [],
              "human_benchmark_eligible": False, "planned_pairs": len(selected)}
    with batch_lock(destination):
        plan = destination / "job.json"
        if plan.exists():
            if read_json(plan) != manifest:
                raise KnowledgeError("Automatic job directory belongs to different immutable inputs")
        else:
            write_private_json(plan, manifest)

        def finish(status, reason=None):
            report.update(status=status, stop_reason=reason)
            write_private_json(destination / ("result-" + str(time.time_ns()) + ".json"), report)
            return report

        # Preflight BOTH models before the first paid call. It has no DB writes.
        common = ["judge", "--input", str(input_path), "--pilot-plan", str(pilot_plan),
                  "--offsets-file", str(queue_path), "--limit", str(limit), "--prompt-variant", prompt_variant,
                  "--max-cost-rub", str(max_cost_rub), "--max-seconds", str(max_seconds)]
        stages = [("primary", primary_model), ("verifier", verifier_model)]
        for name, model in stages:
            result = step([*common, "--model-id", model, "--batch-id", batch_prefix + "-" + name,
                           "--output", str(destination / (name + "-preflight-" + str(time.time_ns()))), "--dry-run"])
            report["steps"].append({"name": name + "_preflight", "result": result})
            if result.get("status") != "dry_run" or not result.get("wire_bound_checked"):
                return finish("stopped", name + "_preflight_failed")
        if not live:
            return finish("dry_run")
        for name, model in stages:
            result = step([*common, "--model-id", model, "--batch-id", batch_prefix + "-" + name,
                "--output", str(destination / name), "--live", "--shared-ledger", "--continue-settled-validation-errors"])
            report["steps"].append({"name": name, "result": result})
            if (result.get("status") not in {"completed", "completed_with_errors"}
                    or result.get("unknown_cost_calls") or result.get("outstanding_reserve_micros")
                    or result.get("unaccounted_attempts")):
                return finish("stopped", name + ":" + str(result.get("stop_reason") or result.get("error")))
        # Exported datasets are create-only. On resume they are regenerated in
        # a fresh derived directory while the paid batch IDs stay unchanged.
        derived = destination / ("derived-" + str(time.time_ns()))
        for name, _ in stages:
            result = step(["export-labels", "--batch-dir", str(destination / name),
                           "--output", str(derived / (name + ".jsonl"))])
            report["steps"].append({"name": name + "_export", "result": result})
            if result.get("status") == "failed" or not (derived / (name + ".jsonl")).is_file():
                return finish("stopped", name + "_export_failed")
        result = step(["machine-consensus", "--primary", str(derived / "primary.jsonl"),
            "--verifier", str(derived / "verifier.jsonl"), "--policy", "binary", "--output-dir", str(derived / "silver")])
        report["steps"].append({"name": "consensus", "result": result})
        if result.get("status") == "failed" or not (derived / "silver" / "silver.jsonl").is_file():
            return finish("stopped", "consensus_failed")
        report["silver"] = str(derived / "silver" / "silver.jsonl")
        if train_shadow:
            rows = read_dataset(report["silver"])
            stories = {value: {r["article"]["cluster_id"] for r in rows if (r["label"]["grade"] >= 4) == value}
                       for value in (True, False)}
            if min(len(v) for v in stories.values()) < 3:
                report["training"] = {"status": "not_trained", "reason": "fewer_than_three_independent_stories_per_class",
                                      "positive_stories": len(stories[True]), "negative_stories": len(stories[False])}
            else:
                report["training"] = step(["train", "--input", report["silver"], "--allow-teacher",
                    "--model-dir", os.getenv("E5_MODEL_DIR", "data/models/e5-base"),
                    "--model-manifest", os.getenv("E5_MODEL_MANIFEST", "knowledge/embedding-model.json"),
                    "--output-dir", str(derived / "model")])
        return finish("completed")
