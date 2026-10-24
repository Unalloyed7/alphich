"""Unattended job must preflight both models and stop after unknown accounting."""
from pathlib import Path
from unittest.mock import patch

import pytest

from alphich_kb.common import KnowledgeError
from alphich_kb.relevance_ml.automatic_job import run_automatic_job


def job(tmp_path, step, **kwargs):
    for name in ("train.jsonl", "manifest.json", "queue.json"):
        path = tmp_path / name
        if not path.exists():
            path.write_text("{}")
    with patch("alphich_kb.relevance_ml.automatic_job.validate_labeling_input"), \
         patch("alphich_kb.relevance_ml.automatic_job.queued_rows", return_value=[{"pair_id": "one"}]):
        return run_automatic_job(input_path=tmp_path / "train.jsonl", pilot_plan=tmp_path / "manifest.json",
            queue_path=tmp_path / "queue.json", output_dir=tmp_path / "job", batch_prefix="test-stable",
            primary_model="z-ai/glm-5.3-flash", verifier_model="qwen/qwen3.6-35b-a3b", step=step, **kwargs)


def test_dry_job_preflights_both_models_without_live_or_budget_flags(tmp_path):
    seen = []

    def step(args):
        seen.append(args)
        return {"status": "dry_run", "wire_bound_checked": True}

    result = job(tmp_path, step)
    assert result["status"] == "dry_run" and len(seen) == 2
    assert all("--dry-run" in args and "--live" not in args for args in seen)


def test_failed_second_preflight_prevents_first_paid_dispatch(tmp_path):
    seen = []

    def step(args):
        seen.append(args)
        return {"status": "dry_run", "wire_bound_checked": len(seen) == 1}

    result = job(tmp_path, step, live=True, shared_ledger=True)
    assert result["status"] == "stopped" and len(seen) == 2
    assert all("--live" not in args for args in seen)


def test_unknown_primary_outcome_never_dispatches_verifier(tmp_path):
    seen = []

    def step(args):
        seen.append(args)
        if "--dry-run" in args:
            return {"status": "dry_run", "wire_bound_checked": True}
        return {"status": "stopped", "stop_reason": "unknown_outcome_requires_manual_reconciliation",
                "unknown_cost_calls": 1, "outstanding_reserve_micros": 100}

    result = job(tmp_path, step, live=True, shared_ledger=True)
    assert result["status"] == "stopped" and len(seen) == 3
    assert "--model-id" in seen[-1] and "z-ai/glm-5.3-flash" in seen[-1]


def test_job_resume_rejects_changed_identity_and_requires_shared_ledger(tmp_path):
    step = lambda args: {"status": "dry_run", "wire_bound_checked": True}
    job(tmp_path, step)
    with pytest.raises(KnowledgeError, match="different immutable"):
        job(tmp_path, step, max_cost_rub="4")
    with pytest.raises(KnowledgeError, match="shared ledger"):
        job(tmp_path, step, live=True)


def test_stopped_job_has_nonzero_exit_for_scheduler_monitoring(capsys):
    from alphich_kb.relevance_ml.cli import main
    with patch("alphich_kb.relevance_ml.automatic_job.run_automatic_job", return_value={"status": "stopped"}):
        code = main(["automatic-job", "--input", "train.jsonl", "--pilot-plan", "manifest.json",
                     "--offsets-file", "queue.json", "--output-dir", "output", "--batch-prefix", "fixed"])
    assert code == 2 and '"stopped"' in capsys.readouterr().out
