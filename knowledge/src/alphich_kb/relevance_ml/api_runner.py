"""Bounded API teacher batches with immutable checkpoints and cost receipts.

Completions propose training labels; they do not update a generative model's
weights. The existing backend Provider owns dispatch and atomic team accounting.
"""
from __future__ import annotations

import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path

from ..common import KnowledgeError, read_json
from .dataset import validate_pair, write_private_json
from .teacher import judge_pair, prepare_judgement, teacher_run_id, validate_checkpoint


def batch_plan(rows, *, batch_id, model_id, model=None):
    """Preflight the entire bounded prefix before the first paid dispatch."""
    if not isinstance(batch_id, str) or not batch_id.strip() or len(batch_id) > 200:
        raise KnowledgeError("Explicit immutable batch_id is required")
    if not isinstance(model_id, str) or not model_id.strip() or len(model_id) > 160:
        raise KnowledgeError("Explicit allowed API model_id is required")
    rows = [validate_pair(row) for row in rows]
    if not rows or len(rows) > 100:
        raise KnowledgeError("API teacher batch needs 1..100 pairs")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise KnowledgeError("Duplicate teacher pair")
    if any(row.get("label", {}).get("origin") == "human" for row in rows):
        raise KnowledgeError("Teacher batching must not overwrite human-reviewed labels")
    plans = [prepare_judgement(row) for row in rows]
    if len({(plan["prompt_version"], plan["prompt_hash"]) for plan in plans}) != 1:
        raise KnowledgeError("A teacher batch must use one prompt contract; separate legacy and enriched company snapshots")
    manifest = {"batch_id": batch_id, "model_id": model_id,
                "prompt_version": plans[0]["prompt_version"], "prompt_hash": plans[0]["prompt_hash"],
                "requests": [plan["input_hash"] for plan in plans]}
    requests = []
    for row, plan in zip(rows, plans):
        wire_bytes = None
        if model is not None:
            from app.intelligence.provider import input_bytes
            wire_bytes = input_bytes(model, plan["system"], plan["data"], plan["response_schema"])
            if wire_bytes > model.input_limit:
                raise KnowledgeError("Teacher request exceeds the configured input bound; pair stays unlabeled")
        requests.append({"pair_id": row["pair_id"], "input_hash": plan["input_hash"],
                         "run_id": teacher_run_id(plan, batch_id, model_id),
                         "article_id": row["article"]["article_id"], "cluster_id": row["article"]["cluster_id"],
                         "source_version": row["article"]["source_version"],
                         "profile_version": row["company"]["profile_version"], "as_of": row["as_of"],
                         "role": row["role"], "article_characters": len(row["article"]["text"]),
                         "request_bytes": wire_bytes,
                         "short_text_review_required": len(row["article"]["text"]) < 200})
    return manifest, {"schema_version": "alphich-api-teacher-preflight-v1", **manifest,
                      "pairs": requests, "paid_calls": 0,
                      "wire_bound_checked": model is not None,
                      "labels_are": "teacher_proposals_not_human_gold"}


def cost_stop_micros(value):
    """A between-call stop, never a promise about an unknown next receipt."""
    if value is None:
        return None
    try:
        amount = Decimal(str(value))
        if not amount.is_finite() or not 0 < amount <= 500:
            raise ValueError()
        return int(amount * 1_000_000)
    except (ValueError, InvalidOperation, OverflowError):
        raise KnowledgeError("--max-cost-rub must be positive and at most 500; it stops between calls") from None


def read_call_ledger(factory, run_id):
    """Copy only accounting fields, without stored output, input or credentials."""
    from sqlalchemy import select
    from app.intelligence.storage import CostReceipt, LLMCall
    with factory() as db:
        calls = list(db.scalars(select(LLMCall).where(LLMCall.run_id == run_id)))
        results = []
        for call in calls:
            receipt = db.get(CostReceipt, call.id)
            results.append({"call_id": call.id, "run_id": call.run_id, "day": call.day, "model_id": call.model,
                            "status": call.status, "error": call.error, "request_hash": call.request_hash,
                            "input_tokens": call.input_tokens, "output_tokens": call.output_tokens,
                            "cost_micros": call.cost_micros, "reserved_micros": call.reserved_micros,
                            "outstanding_reserve_micros": call.reserved_micros if call.cost_micros is None else 0,
                            "receipt": None if receipt is None else {
                                "amount": receipt.amount, "rub_per_unit": receipt.rub_per_unit, "source": receipt.source}})
        return results


@contextmanager
def batch_lock(destination):
    # flock releases automatically on a crash, unlike a create-only lock file.
    import fcntl
    destination.mkdir(parents=True, exist_ok=True, mode=0o700)
    destination.chmod(0o700)
    with (destination / ".runner.lock").open("a+", encoding="utf-8") as lock:
        Path(lock.name).chmod(0o600)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise KnowledgeError("This teacher batch is already running") from None
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def _latency(values, percentile):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(len(ordered) * percentile) - 1)]


def summarize_batch(rows, destination, manifest, ledger_reader):
    """Refresh authoritative receipt totals, including failures and reconciliations."""
    completed = failures = 0
    completed_runs = set()
    calls, latencies = [], []
    for row in rows:
        plan = prepare_judgement(row)
        run_id = teacher_run_id(plan, manifest["batch_id"], manifest["model_id"])
        calls.extend(ledger_reader(run_id))
        result_path = destination / (row["pair_id"] + ".json")
        if result_path.exists():
            validate_checkpoint(read_json(result_path), manifest)
            completed += 1
            completed_runs.add(run_id)
        if (destination / (row["pair_id"] + ".error.json")).exists():
            failures += 1
        audit_path = destination / "audit" / (row["pair_id"] + ".json")
        if audit_path.exists():
            audit = read_json(audit_path)
            if audit.get("run_id") != run_id or audit.get("input_hash") != plan["input_hash"]:
                raise KnowledgeError("Teacher audit sidecar belongs to different immutable inputs")
            if isinstance(audit.get("elapsed_seconds"), (int, float)):
                latencies.append(audit["elapsed_seconds"])
    # Ledger identities are unique even if an interrupted process resumed.
    calls = list({call["call_id"]: call for call in calls}.values())
    cost = sum(call["cost_micros"] for call in calls if call.get("cost_micros") is not None)
    reserve = sum(call.get("outstanding_reserve_micros", 0) for call in calls)
    unknown = sum(call.get("cost_micros") is None for call in calls)
    unaccounted = len(completed_runs - {call["run_id"] for call in calls})
    return {"schema_version": "alphich-api-teacher-summary-v1", "batch_id": manifest["batch_id"],
            "model_id": manifest["model_id"], "prompt_hash": manifest["prompt_hash"],
            "total_pairs": len(rows), "completed": completed, "unlabeled_failures": failures,
            "pending": len(rows) - completed - failures, "accounted_calls": len(calls),
            "known_cost_micros": cost, "known_cost_rub": str(Decimal(cost) / 1_000_000),
            "unknown_cost_calls": unknown, "outstanding_reserve_micros": reserve,
            "unaccounted_completed_labels": unaccounted,
            "cost_per_accepted_label_rub": str(Decimal(cost) / 1_000_000 / completed) if completed and not unknown and not unaccounted else None,
            "input_tokens": sum(call.get("input_tokens") or 0 for call in calls),
            "output_tokens": sum(call.get("output_tokens") or 0 for call in calls),
            "latency_p50_seconds": _latency(latencies, .5), "latency_p95_seconds": _latency(latencies, .95),
            "usage_incomplete_calls": sum(call.get("input_tokens") is None or call.get("output_tokens") is None for call in calls),
            "shared_budget_accounted": not unaccounted, "calls": calls,
            "labels_are": "teacher_proposals_not_human_gold", "latency_scope": "locally_observed_attempts"}


def run_api_batch(rows, provider, *, destination, batch_id, model_id, ledger_reader,
                  max_cost_rub=None, max_seconds=1200, clock=time.monotonic):
    """Sequential pilot/batch; no auto retry after invalid/unknown/HTTP failures."""
    if type(max_seconds) is not int or not 1 <= max_seconds <= 7200:
        raise KnowledgeError("API teacher runtime bound must be 1..7200 seconds")
    rows = [validate_pair(row) for row in rows]
    stop_cost = cost_stop_micros(max_cost_rub)
    model = provider.settings.models[model_id] if model_id in provider.settings.models else None
    if model is None:
        raise KnowledgeError("Teacher model is not in the configured allowed catalog")
    manifest, preflight = batch_plan(rows, batch_id=batch_id, model_id=model_id, model=model)
    destination = Path(destination)
    started = clock()
    with batch_lock(destination):
        manifest_path = destination / "batch.json"
        if manifest_path.exists():
            if read_json(manifest_path) != manifest:
                raise KnowledgeError("Batch directory is already bound to different immutable inputs")
        else:
            write_private_json(manifest_path, manifest)
        audit_dir = destination / "audit"
        audit_dir.mkdir(exist_ok=True, mode=0o700)
        preflight_path = audit_dir / "preflight.json"
        if not preflight_path.exists():
            write_private_json(preflight_path, preflight)
        ledger_cache = {}
        def cached_ledger(run_id):
            if run_id not in ledger_cache:
                ledger_cache[run_id] = ledger_reader(run_id)
            return ledger_cache[run_id]
        reason = None
        for row in rows:
            result_path = destination / (row["pair_id"] + ".json")
            if result_path.exists():
                validate_checkpoint(read_json(result_path), manifest)
                continue
            error_path = destination / (row["pair_id"] + ".error.json")
            if error_path.exists():
                continue
            summary = summarize_batch(rows, destination, manifest, cached_ledger)
            if summary["unaccounted_completed_labels"]:
                reason = "completed_label_missing_accounting_record"
                break
            if summary["unknown_cost_calls"] or summary["outstanding_reserve_micros"]:
                reason = "unknown_outcome_requires_manual_reconciliation"
                break
            if stop_cost is not None and summary["known_cost_micros"] >= stop_cost:
                reason = "between_call_cost_stop"
                break
            if clock() - started >= max_seconds:
                reason = "runtime_stop_before_dispatch"
                break
            plan = prepare_judgement(row)
            run_id = teacher_run_id(plan, batch_id, model_id)
            stamp = datetime.now(timezone.utc).isoformat()
            call_started = clock()
            error_code = None
            validation_error = None
            try:
                result = judge_pair(row, provider, batch_id=batch_id, model_id=model_id)
            except Exception as error:
                if isinstance(error, KnowledgeError):
                    error_code, validation_error = "teacher_validation_failed", str(error)[:400]
                else:
                    error_code = getattr(error, "code", None) or type(error).__name__
                if error_code != "budget_exceeded":
                    write_private_json(error_path, {"pair_id": row["pair_id"], "run_id": run_id,
                                       "status": "unlabeled", "error": error_code,
                                       "validation_error": validation_error})
                reason = error_code
            else:
                write_private_json(result_path, result)
            ledger_cache[run_id] = ledger_reader(run_id)
            audit_path = audit_dir / (row["pair_id"] + ".json")
            # A reservation rejection has no billed attempt. Keep it resumable
            # under the original identity without claiming an API response.
            if error_code != "budget_exceeded" and not audit_path.exists():
                write_private_json(audit_path, {"pair_id": row["pair_id"], "input_hash": plan["input_hash"],
                    "batch_id": batch_id, "model_id": model_id, "prompt_hash": plan["prompt_hash"],
                    "run_id": run_id, "started_at": stamp, "elapsed_seconds": round(clock() - call_started, 6),
                    "status": "accepted_teacher_label" if error_code is None else "unlabeled", "error": error_code,
                    "validation_error": validation_error,
                    "calls": ledger_cache[run_id]})
            if error_code is not None:
                break
        summary = summarize_batch(rows, destination, manifest, ledger_reader)
        if summary["unaccounted_completed_labels"]:
            reason = "completed_label_missing_accounting_record"
        summary.update(status="completed" if summary["completed"] == len(rows) and not reason else "stopped",
                       stop_reason=reason, elapsed_seconds=round(clock() - started, 6),
                       max_cost_rub=str(max_cost_rub) if max_cost_rub is not None else None,
                       cost_stop_semantics="between_calls_only; next receipt can exceed this value; shared daily cap is hard")
        summary_path = audit_dir / ("summary-" + str(time.time_ns()) + ".json")
        write_private_json(summary_path, summary)
        return {**summary, "audit_summary": str(summary_path)}
