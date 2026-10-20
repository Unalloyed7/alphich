"""Opt-in live development checks through the real API, workers and retriever.

Synthetic company profiles and explicitly labelled controls. Automated assertions
measure contract/provenance; semantic quality still requires independent review.
"""
from __future__ import annotations

import json
import os
import re
import time
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from uuid import uuid4

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core import Base, Company, import_companies, make_session_factory
from app.intelligence.provider import hashed
from app.news import Job
from app.worker import process_next_job
from .provider import canonical
from .service import Runtime
from .settings import AnalysisError
from .storage import BudgetDay, LLMCall
from .worker import process_next


REPORT_VERSION = "llm-development-report-v2"
# Reports use a closed vocabulary: an upstream exception is never a diagnostic.
SAFE_ERRORS = frozenset("""analysis_failed analysis_lease_lost analysis_task_missing analysis_lease_expired
analysis_cache_lease_lost analysis_in_progress analysis_outcome_unknown extraction_in_progress
extraction_outcome_unknown extraction_lease_lost source_version_changed insufficient_news_facts
parent_insight_unavailable event_outside_scope event_unavailable_as_of token_prices_unavailable
invalid_llm_configuration llm_not_configured resumed_call_context_changed previous_call_outcome_unknown
agent_call_limit budget_exceeded model_not_allowed context_too_large llm_timeout_outcome_unknown
llm_response_too_large llm_incomplete_response llm_usage_missing provider_exceeded_declared_token_limits
provider_exceeded_reserved_budget llm_invalid_response llm_cost_receipt_missing usage_unknown_reserve_retained
cost_unknown_reserve_retained llm_invalid_json llm_schema_invalid llm_network_outcome_unknown llm_network_error invalid_extraction_review
repeated_tool_without_new_evidence evidence_outside_scope wrong_evidence_kind agent_tool_limit
card_validation_failed evaluation_job_failed evaluation_worker_did_not_claim_job evaluation_job_not_completed
evaluation_requires_dedicated_synthetic_database evaluation_database_has_pending_jobs
evaluation_internal_error evaluation_fixture_company_mismatch""".split())
SAFE_ERRORS = SAFE_ERRORS | {"invalid_research_handoff", "research_agent_limit", "invalid_verification_review"}


def safe_error(value):
    if isinstance(value, str) and (value in SAFE_ERRORS or re.fullmatch(r"(?:llm|evaluation_api)_http_[1-5][0-9]{2}", value)):
        return value
    return "evaluation_internal_error"


def _validate_cases(data):
    if not isinstance(data, dict):
        raise ValueError()
    sources, companies, roles = data["sources"], data["companies"], data["roles"]
    if (not isinstance(sources, list) or not sources or not isinstance(companies, list) or not companies or
            not isinstance(roles, list) or not roles or len(roles) != len(set(roles)) or set(roles) - {"cfo", "ceo"} or
            len(companies) != len(set(companies)) or
            any(not isinstance(c, str) or not re.fullmatch(r"LLM_DEMO_[A-Z0-9_]+", c) for c in companies) or
            not isinstance(data["version"], str) or not data["version"] or not isinstance(data["scope"], str)):
        raise ValueError()
    ids = []
    for source in sources:
        if (not isinstance(source, dict) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,79}", source["id"]) or
                any(not isinstance(source[k], str) or not source[k] for k in ("title", "text", "provenance"))):
            raise ValueError()
        ids.append(source["id"])
        for key in ("minimum_facts", "maximum_facts"):
            if key in source and (type(source[key]) is not int or not 0 <= source[key] <= 10):
                raise ValueError()
        if source.get("maximum_facts", 10) < source.get("minimum_facts", 0):
            raise ValueError()
        for key in ("expected_topics", "forbidden_output", "require_personal_card_for", "expected_relevance_codes", "forbidden_relevance_codes"):
            if key in source and (not isinstance(source[key], list) or any(not isinstance(v, str) for v in source[key])):
                raise ValueError()
        if set(source.get("require_personal_card_for", [])) - set(companies):
            raise ValueError()
        if source.get("expected_status") not in (None, "insufficient_evidence", "ready_with_conditions"):
            raise ValueError()
        for key in ("forbid_instruction_facts", "forbid_personal_inference"):
            if type(source.get(key, False)) is not bool:
                raise ValueError()
        if not isinstance(source.get("expected_fact_attributes", {}), dict):
            raise ValueError()
        features = source.get("required_personal_feature_by_company", {})
        if not isinstance(features, dict) or set(features) - set(companies) or any(not isinstance(v, str) for v in features.values()):
            raise ValueError()
    if len(ids) != len(set(ids)):
        raise ValueError()
    return data


def read_cases(path):
    try:
        data = json.loads(Path(path).read_text())
        return _validate_cases(data)
    except (OSError, ValueError, KeyError, TypeError):
        raise AnalysisError("invalid_evaluation_cases") from None


def check_result(card, event, source, evidence, forbidden=()):
    """No model grades itself here: exact quotes, IDs and boundaries only."""
    issues = []
    if not all(isinstance(v, dict) for v in (card, event, source, evidence)):
        return ["invalid_result_structure"]
    if (not isinstance(source.get("text"), str) or
            any(not isinstance(card.get(k, []), list) for k in ("claims", "risks", "opportunities", "next_steps", "missing_data")) or
            not isinstance(card.get("bank_view", {}), dict)):
        return ["invalid_result_structure"]
    facts, raw_claims = event.get("facts", []), card.get("claims", [])
    if not isinstance(facts, list) or not isinstance(raw_claims, list):
        return ["invalid_result_structure"]
    for fact in facts:
        if not isinstance(fact, dict) or not isinstance(fact.get("source_span"), dict):
            issues.append("news_quote_span_mismatch")
            continue
        span = fact.get("source_span", {})
        start, end = span.get("start"), span.get("end")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(source["text"]) or
                source["text"][start:end] != fact.get("text")):
            issues.append("news_quote_span_mismatch")
        if source.get("forbid_instruction_facts") and any(marker in str(fact.get("text", "")) for marker in forbidden):
            issues.append("injection_instruction_in_extracted_fact")
    if any(not isinstance(c, dict) or not isinstance(c.get("id"), str) or not isinstance(c.get("text"), str) or
           not isinstance(c.get("conditions", []), list) or any(not isinstance(v, str) for v in c.get("conditions", [])) for c in raw_claims):
        return sorted(set(issues + ["invalid_result_structure"]))
    claims = {c["id"]: c for c in raw_claims}
    if len(claims) != len(raw_claims):
        issues.append("duplicate_claim_ids")
    fact_kinds = {"news_fact": "event_fact", "company_fact": "company_feature", "bank_view": "bank_claim"}
    for claim in claims.values():
        refs = claim.get("evidence_ids")
        if not isinstance(refs, list) or not refs or any(not isinstance(i, str) or not isinstance(evidence.get(i), dict) for i in refs):
            issues.append("unknown_evidence_id")
            continue
        if len(set(refs)) != len(refs):
            issues.append("duplicate_evidence_ids")
        kinds = {evidence[i].get("kind") for i in refs}
        if any(evidence[i].get("company_id") != card.get("company_id") for i in refs
               if evidence[i].get("kind") in {"event_fact", "company_feature"}):
            issues.append("evidence_company_mismatch")
        if claim.get("kind") == "model_inference":
            if not {"event_fact", "company_feature"} <= kinds or not claim.get("conditions"):
                issues.append("inference_missing_personal_basis")
            if any(ch.isdigit() for ch in str(claim.get("text", ""))):
                issues.append("number_in_model_inference")
        elif claim.get("kind") not in fact_kinds:
            issues.append("unknown_claim_kind")
        elif not any(evidence[i].get("text") == claim.get("text") and evidence[i].get("kind") == fact_kinds[claim["kind"]] for i in refs):
            issues.append("factual_claim_not_exact_evidence")
        if claim.get("kind") == "bank_view" and any(evidence[i].get("position_role") != "current_in_document" for i in refs if evidence[i].get("kind") == "bank_claim"):
            issues.append("bank_view_not_current_reviewed_claim")
    for ref in card.get("risks", []) + card.get("opportunities", []):
        if not isinstance(ref, str) or claims.get(ref, {}).get("kind") != "model_inference":
            issues.append("risk_or_opportunity_missing_inference")
    for step in card.get("next_steps", []):
        if (not isinstance(step, dict) or not isinstance(step.get("conditions"), list) or not step["conditions"] or
                not isinstance(step.get("basis_claim_ids"), list) or not step["basis_claim_ids"] or
                any(not isinstance(ref, str) or claims.get(ref, {}).get("kind") != "model_inference" for ref in step.get("basis_claim_ids", []))):
            issues.append("action_missing_inference_basis_or_conditions")
    if card.get("bank_view", {}).get("status") == "not_found" and any(c.get("kind") == "bank_view" for c in raw_claims):
        issues.append("bank_view_status_contradicts_claims")
    # Factual text is copied by the server: quoting a marker is not evidence of obedience.
    # Contaminated extraction is reported separately above. Inspect only free model text here.
    generated = canonical({"inferences": [c.get("text") for c in raw_claims if c.get("kind") == "model_inference"],
                           "conditions": [c.get("conditions", []) for c in raw_claims],
                           "next_steps": card.get("next_steps", []), "missing_data": card.get("missing_data", [])})
    if any(marker in generated for marker in forbidden):
        issues.append("injection_marker_in_generated_output")
    return sorted(set(issues))


def check_case(row, source, with_questions=False):
    """Reproducible contract checks; these do not establish semantic correctness."""
    card, event, evidence = row.get("card"), row.get("event", {"facts": row.get("facts", [])}), row.get("evidence", {})
    if not isinstance(card, dict):
        return ["analysis_result_missing"]
    issues = check_result(card, event, source, evidence, source.get("forbidden_output", []))
    facts = event.get("facts", [])
    if "invalid_result_structure" in issues or not isinstance(facts, list):
        return sorted(set(issues + ["invalid_result_structure"]))
    if len(facts) < source.get("minimum_facts", 0):
        issues.append("too_few_verified_facts")
    if len(facts) > source.get("maximum_facts", 10):
        issues.append("unexpected_extracted_facts")
    attrs = [f["attributes"] for f in facts if isinstance(f, dict) and isinstance(f.get("attributes"), dict)]
    topics = {t for a in attrs for t in a.get("topics", []) if isinstance(t, str)}
    if not set(source.get("expected_topics", [])) <= topics:
        issues.append("expected_topic_missing")
    expected = source.get("expected_fact_attributes", {})
    if expected and not any(all(a.get(k) == v for k, v in expected.items()) for a in attrs):
        issues.append("expected_fact_attributes_missing")
    if source.get("expected_status") and card.get("status") != source["expected_status"]:
        issues.append("unexpected_card_status")
    if source.get("forbid_personal_inference") and any(c.get("kind") == "model_inference" for c in card.get("claims", [])):
        issues.append("unexpected_personal_inference")
    reasons = card.get("relevance_reasons", [])
    if not isinstance(reasons, list):
        issues.append("invalid_result_structure")
        codes = set()
    else:
        codes = {reason["code"] for reason in reasons if isinstance(reason, dict) and isinstance(reason.get("code"), str)}
    if not set(source.get("expected_relevance_codes", [])) <= codes:
        issues.append("expected_relevance_reason_missing")
    if set(source.get("forbidden_relevance_codes", [])) & codes:
        issues.append("forbidden_relevance_reason")
    if row["company_id"] in source.get("require_personal_card_for", []):
        if card.get("status") != "ready_with_conditions" or not any(c.get("kind") == "model_inference" for c in card.get("claims", [])):
            issues.append("expected_personal_card_missing")
    feature = source.get("required_personal_feature_by_company", {}).get(row["company_id"])
    if feature and not any(c.get("kind") == "model_inference" and any(
            evidence.get(ref, {}).get("kind") == "company_feature" and evidence[ref].get("name") == feature and evidence[ref].get("value") is True
            for ref in c.get("evidence_ids", [])) for c in card.get("claims", [])):
        issues.append("expected_personal_product_basis_missing")
    if (card.get("company_id"), card.get("role")) != (row["company_id"], row["role"]):
        issues.append("company_or_role_changed")
    if row.get("event_id") and card.get("event_id") != row["event_id"]:
        issues.append("analysis_event_changed")
    if with_questions and source.get("question") and facts:
        answer = row.get("answer")
        if not isinstance(answer, dict):
            issues.append("requested_question_not_completed")
        else:
            issues += check_result(answer, event, source, row.get("answer_evidence", {}), source.get("forbidden_output", []))
            if any(answer.get(key) != card.get(key) for key in ("company_id", "role", "event_id", "event_version", "as_of", "profile_version")) or answer.get("insight_id") != card.get("id"):
                issues.append("question_context_changed")
            if answer.get("question") != source["question"]:
                issues.append("question_text_changed")
    return sorted(set(issues))


def summarize_report(report):
    rows = report["results"]
    completed = len(rows)
    passed = sum(r.get("status") == "passed_controls" for r in rows)
    failed = completed - passed
    stopped = completed < report["planned_cases"]
    full_count = report.get("full_case_count", report["planned_cases"])
    selected_count = report.get("filtered_case_count", report["planned_cases"])
    required = [r for r in rows if r.get("expected_personal_card")]
    ready = sum(r.get("card", {}).get("status") == "ready_with_conditions" and any(c.get("kind") == "model_inference" for c in r.get("card", {}).get("claims", [])) for r in required)
    return {"completed_cases": completed, "passed_control_cases": passed, "failed_cases": failed,
            "execution_failures": sum(r.get("status") == "failed" for r in rows),
            "stopped_early": stopped, "not_run_cases": report["planned_cases"] - completed,
            "selected_suite_complete": not stopped,
            "full_suite_complete": not stopped and report["planned_cases"] == full_count,
            "limited_selection": report["planned_cases"] < selected_count,
            "source_filter_applied": selected_count < full_count,
            "all_selected_controls_passed": bool(completed) and not failed and not stopped,
            "personal_cards": {"required_in_attempted_cases": len(required), "ready": ready, "missing": len(required) - ready},
            "semantic_quality_independently_reviewed": False}


def recheck_report(report_or_path, cases_path=None):
    """Re-evaluate saved artifacts without API, database access or model calls."""
    try:
        report = deepcopy(report_or_path) if isinstance(report_or_path, dict) else json.loads(Path(report_or_path).read_text())
        cases = read_cases(cases_path) if cases_path else _validate_cases(report["case_manifest"])
        if hashed(cases) != report["case_hash"]:
            raise AnalysisError("evaluation_case_manifest_changed")
        sources = {s["id"]: s for s in cases["sources"]}
        for row in report["results"]:
            if row.get("status") == "failed":
                row["error"] = safe_error(row.get("error"))
                continue  # Missing artifacts never turn an execution failure into a pass.
            source_id = row.get("source_id", row["case_id"].split(":", 1)[0])
            row["issues"] = check_case(row, sources[source_id], report.get("with_questions", False))
            row["status"] = "failed_controls" if row["issues"] else "passed_controls"
        return {"report_version": REPORT_VERSION, "rechecked_without_model_calls": True,
                **summarize_report(report), "results": [{k: r.get(k) for k in ("case_id", "status", "issues", "error")} for r in report["results"]]}
    except (OSError, ValueError, KeyError, TypeError):
        raise AnalysisError("invalid_evaluation_report") from None


def _rub(micros):
    return str(Decimal(micros) / Decimal(1_000_000))


def _safe_run(run):
    """Only bounded audit metadata; no request/result bodies or exception repr."""
    return {"id": run["id"], "status": run["status"], "company_id": run["company_id"], "role": run["role"],
            "versions": run.get("versions", {}),
            "verification": {"status": run.get("verification", {}).get("status"),
                             "issues": [safe_error(v) for v in run.get("verification", {}).get("issues", [])]},
            "tool_actions": [s.get("action") for s in run.get("steps", []) if s.get("action") in {
                "search_bank_materials", "search_related_events", "get_company_features", "get_payment_neighbors", "get_evidence", "cache_hit"}],
            "agent_actions": [{"agent_role": s.get("agent_role"), "action": s.get("action"), "to_agent": s.get("to_agent")}
                              for s in run.get("steps", []) if s.get("agent_role") in {"analyst", "bank_researcher", "verifier", "coordinator"}
                              and s.get("action") in {"search_bank_materials", "search_related_events", "get_company_features",
                                                      "get_payment_neighbors", "get_evidence", "agent_handoff", "verify_card", "verify_card_schema"}],
            "llm_calls": [{k: c.get(k) for k in ("id", "model", "purpose", "agent_role", "status", "input_tokens", "output_tokens", "cost_rub", "cost_source", "reserved_rub")} |
                          {"error": safe_error(c["error"]) if c.get("error") else None} for c in run.get("llm_calls", [])]}


def evaluate(settings, cases_path, company_fixture, limit, output, with_questions=False, *, runtime_factory=Runtime, source_ids=()):
    from app.api import create_app
    if type(limit) is not int or not 1 <= limit <= 100:
        raise AnalysisError("invalid_evaluation_limit")
    cases = read_cases(cases_path)
    if set(source_ids) - {s["id"] for s in cases["sources"]}:
        raise AnalysisError("unknown_evaluation_source")
    sources = [s for s in cases["sources"] if not source_ids or s["id"] in source_ids]
    filtered = [(source, company, role) for source in sources for company in cases["companies"] for role in cases["roles"]]
    selected = filtered[:limit]
    output = Path(output)
    # Reports contain publication text and analysis. Keep them in ignored local data.
    local_root = Path("data/processed").resolve()
    if not output.resolve().is_relative_to(local_root):
        raise AnalysisError("evaluation_output_must_be_local_processed_data")
    output.parent.mkdir(parents=True, exist_ok=True)
    engine, factory = make_session_factory()
    Base.metadata.create_all(engine)
    report = {"report_version": REPORT_VERSION, "run_id": "eval_" + uuid4().hex, "started_at": datetime.now(timezone.utc).isoformat(),
              "case_version": cases["version"], "case_hash": hashed(cases), "scope": cases["scope"],
              "case_manifest": cases, "with_questions": with_questions,
              "synthetic_companies": True, "semantic_quality_independently_reviewed": False,
              "limitations": ["open_development_controls_not_independent_benchmark", "short_excerpts_not_full_news_corpus",
                              "automated_contract_checks_do_not_prove_semantic_quality", "synthetic_profiles_not_customer_quality_evaluation"],
              "budget_mode": settings.budget_mode, "receipt_rub_per_unit": str(settings.receipt_rub_per_unit),
              "full_case_count": len(cases["sources"]) * len(cases["companies"]) * len(cases["roles"]),
              "filtered_case_count": len(filtered), "planned_cases": len(selected),
              "selected_case_ids": [f"{source['id']}:{company}:{role}" for source, company, role in selected],
              "results": []}
    previous_allowlist = os.environ.get("DEMO_COMPANY_IDS")

    def persist():
        temporary = output.with_suffix(".partial")
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        # Defense in depth if a future diagnostic accidentally includes configuration.
        if settings.api_key:
            serialized = serialized.replace(settings.api_key, "[REDACTED_CREDENTIAL]")
        with os.fdopen(os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(serialized)
        temporary.replace(output)

    def require(response, status=200):
        if response.status_code != status:
            # Never copy raw provider errors, headers or credentials into a report.
            raise AnalysisError("evaluation_api_http_" + str(response.status_code))
        return response.json()

    def complete(client, queued, headers, row, label, analysis_runtime=None):
        job_id = queued["id"]
        row.setdefault("jobs", {})[label] = {"id": job_id, "status": "queued"}
        for _ in range(5):
            job = require(client.get("/api/v1/jobs/" + job_id, headers=headers))
            row["jobs"][label] = {k: job.get(k) for k in ("id", "status", "stage", "result_type", "result_id")}
            row["jobs"][label]["error"] = safe_error(job["error"]) if job.get("error") else None
            if analysis_runtime:
                run_id = "run_" + hashed(job_id)[:32]
                response = client.get("/api/v1/runs/" + run_id, headers=headers)
                if response.status_code == 200:
                    row.setdefault("attempt_runs", {})[label] = _safe_run(response.json())
            if job["status"] in {"succeeded", "failed"}:
                if job["status"] != "succeeded":
                    raise AnalysisError(safe_error(job["error"]) if job.get("error") else "evaluation_job_failed")
                return job
            worked = process_next(factory, analysis_runtime) if analysis_runtime else process_next_job(factory)
            if not worked:
                raise AnalysisError("evaluation_worker_did_not_claim_job")
        raise AnalysisError("evaluation_job_not_completed")

    try:
        with factory() as db:
            imported = set(db.scalars(select(Company.id)))
            # Dedicated evaluation DB: never process somebody else's queued jobs or real customers.
            if imported and imported != set(cases["companies"]):
                raise AnalysisError("evaluation_requires_dedicated_synthetic_database")
            if db.scalar(select(Job.id).where(Job.status.in_(["queued", "running"])).limit(1)):
                raise AnalysisError("evaluation_database_has_pending_jobs")
        if not imported:
            import_companies(company_fixture, factory)
        with factory() as db:
            if set(db.scalars(select(Company.id))) != set(cases["companies"]):
                raise AnalysisError("evaluation_fixture_company_mismatch")
        os.environ["DEMO_COMPANY_IDS"] = ",".join(cases["companies"])
        runtime = runtime_factory(factory, settings)
        with TestClient(create_app(factory, engine)) as client:
            for source, company, role in selected:
                started = time.monotonic()
                with factory() as db:
                    before_calls = set(db.scalars(select(LLMCall.id)))
                row = {"case_id": f"{source['id']}:{company}:{role}", "source_id": source["id"], "company_id": company, "role": role,
                       "provenance": source["provenance"], "source_url": source.get("source_url"), "issues": []}
                row["expected_personal_card"] = company in source.get("require_personal_card_for", [])
                row["question_status"] = "pending" if with_questions and source.get("question") else "not_requested"
                report["results"].append(row)
                try:
                    session = require(client.post("/api/v1/demo-session", json={"company_id": company, "role": role}), 201)
                    headers = {"X-Demo-Session": session["token"]}
                    def post(path, body):
                        return require(client.post(path, headers={**headers, "Idempotency-Key": "eval-" + uuid4().hex}, json=body), 202)
                    publication = {k: source[k] for k in ("title", "text", "source_url", "published_at") if k in source}
                    ingested = complete(client, post("/api/v1/publications", publication), headers, row, "ingestion")
                    event_id = ingested["result_id"]
                    row["event_id"] = event_id
                    analysis = complete(client, post(f"/api/v1/companies/{company}/insights", {"event_id": event_id}), headers, row, "analysis", runtime)
                    card = require(client.get("/api/v1/insights/" + analysis["result_id"], headers=headers))
                    event = require(client.get("/api/v1/events/" + event_id, headers=headers))
                    def cited_evidence(payload, answer_id=None):
                        params = {"insight_id": card["id"]}
                        if answer_id:
                            params["answer_id"] = answer_id
                        ids = {i for c in payload["claims"] for i in c["evidence_ids"]}
                        return {i: require(client.get("/api/v1/evidence/" + i, params=params, headers=headers)) for i in ids}
                    evidence = cited_evidence(card)
                    row.update(card=card, event=event, facts=event["facts"], evidence=evidence,
                               run=_safe_run(require(client.get("/api/v1/runs/" + card["run_id"], headers=headers))))
                    row["issues"] = check_case(row, source)
                    if settings.api_key and settings.api_key in canonical({"card": card, "evidence": evidence}):
                        row["issues"].append("credential_in_result")
                    if with_questions and source.get("question") and event["facts"] and not row["issues"]:
                        answer_job = complete(client, post("/api/v1/insights/" + card["id"] + "/questions",
                                                            {"question": source["question"]}), headers, row, "question", runtime)
                        answer = require(client.get("/api/v1/answers/" + answer_job["result_id"], headers=headers))
                        answer_evidence = cited_evidence(answer, answer["id"])
                        row.update(answer=answer, answer_evidence=answer_evidence,
                                   answer_run=_safe_run(require(client.get("/api/v1/runs/" + answer["run_id"], headers=headers))))
                        row["issues"] += check_case(row, source, True)
                        row["question_status"] = "completed"
                    elif row["question_status"] == "pending":
                        row["question_status"] = "not_applicable_no_facts" if not event["facts"] else "skipped_after_card_failure"
                    row["issues"] = sorted(set(row["issues"]))
                    row["status"] = "passed_controls" if not row["issues"] else "failed_controls"
                except AnalysisError as exc:
                    row.update(status="failed", error=safe_error(exc.code))
                except Exception:
                    row.update(status="failed", error="evaluation_internal_error")
                finally:
                    if row["question_status"] == "pending":
                        row["question_status"] = "skipped_after_execution_failure"
                    row["elapsed_seconds"] = round(time.monotonic() - started, 3)
                    with factory() as db:
                        new_calls = [c for c in db.scalars(select(LLMCall)) if c.id not in before_calls]
                        row["usage"] = {"new_calls": len(new_calls),
                            "input_tokens": sum(c.input_tokens or 0 for c in new_calls),
                            "output_tokens": sum(c.output_tokens or 0 for c in new_calls),
                            "cost_rub": _rub(sum(c.cost_micros or 0 for c in new_calls)),
                            "unsettled_calls": sum(c.cost_micros is None for c in new_calls),
                            "unsettled_reservation_rub": _rub(sum(c.reserved_micros for c in new_calls if c.cost_micros is None)),
                            "calls": [{"id": c.id, "run_id": c.run_id, "purpose": c.purpose, "status": c.status,
                                       "error": safe_error(c.error) if c.error else None} for c in new_calls]}
                    report.update(summarize_report(report))
                    persist()
                # Do not burn more of the shared budget after a failure.
                if row["status"] != "passed_controls":
                    break
        with factory() as db:
            calls = db.scalars(select(LLMCall)).all()
            budgets = db.scalars(select(BudgetDay)).all()
            report["ledger_total"] = {"call_count": len(calls), "spent_rub": _rub(sum(b.spent_micros for b in budgets)),
                                      "reserved_rub": _rub(sum(b.reserved_micros for b in budgets)),
                                      "note": "Whole dedicated database, including previous probes/evaluations"}
        report["finished_at"] = datetime.now(timezone.utc).isoformat()
        report.update(summarize_report(report))
        persist()
        return {"report": str(output), **summarize_report(report),
                "scope": report["scope"], "ledger_total": report["ledger_total"]}
    except Exception as exc:
        report.update(summarize_report(report))
        report.update(setup_error=safe_error(exc.code) if isinstance(exc, AnalysisError) else "evaluation_internal_error",
                      finished_at=datetime.now(timezone.utc).isoformat())
        persist()
        raise
    finally:
        if previous_allowlist is None:
            os.environ.pop("DEMO_COMPANY_IDS", None)
        else:
            os.environ["DEMO_COMPANY_IDS"] = previous_allowlist
        engine.dispose()
