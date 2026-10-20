"""Offline controls and failed-worker diagnostics; no live gateway requests."""
import copy
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from app.core import Base, Company, make_session_factory
from app.intelligence.evaluation import check_case, check_result, evaluate, read_cases, recheck_report, summarize_report
from app.intelligence.provider import hashed
from app.intelligence.settings import AnalysisError, Settings
from app.intelligence.storage import AnalysisRun, LLMCall


COMPANY = "LLM_DEMO_LOAN"
TEXT = "Банк России сохранил ключевую ставку."
ROOT = Path(__file__).resolve().parents[2]


def artifact():
    source = {"id": "rate", "title": "Учебное решение о ставке", "text": TEXT, "provenance": "synthetic_test",
              "minimum_facts": 1, "expected_topics": ["policy_rate"],
              "expected_fact_attributes": {"geography": "RU", "kind": "rate_decision"},
              "require_personal_card_for": [COMPANY], "required_personal_feature_by_company": {COMPANY: "CL_PRODUCT_LOA_MARK"}}
    fact = {"id": "fact", "text": TEXT, "source_span": {"start": 0, "end": len(TEXT)},
            "attributes": {"topics": ["policy_rate"], "geography": "RU", "kind": "rate_decision"}}
    evidence = {"fact": {**fact, "kind": "event_fact", "company_id": COMPANY},
                "feature": {"id": "feature", "kind": "company_feature", "company_id": COMPANY,
                            "name": "CL_PRODUCT_LOA_MARK", "value": True, "text": "Кредитный продукт: да."}}
    card = {"id": "insight", "company_id": COMPANY, "role": "cfo", "event_id": "event", "event_version": 2,
            "profile_version": "profile", "as_of": "2026-10-20T00:00:00+00:00", "status": "ready_with_conditions",
            "claims": [{"id": "c1", "kind": "news_fact", "text": TEXT, "evidence_ids": ["fact"], "conditions": []},
                       {"id": "c2", "kind": "model_inference", "text": "При пересмотре ставки договора условия финансирования могут измениться.",
                        "evidence_ids": ["fact", "feature"], "conditions": ["Если договор связан с ключевой ставкой"]}],
            "risks": ["c2"], "opportunities": [],
            "next_steps": [{"text": "Уточнить условия договора", "basis_claim_ids": ["c2"], "conditions": ["Если договор действует"]}],
            "missing_data": ["Условия договора"], "bank_view": {"status": "not_found"}}
    return source, {"case_id": "rate:" + COMPANY + ":cfo", "source_id": "rate", "company_id": COMPANY,
                    "role": "cfo", "event_id": "event", "event": {"facts": [fact]}, "evidence": evidence,
                    "card": card, "expected_personal_card": True, "status": "passed_controls"}


def test_applicable_case_requires_ready_inference_and_its_product_basis():
    source, row = artifact()
    assert check_case(row, source) == []
    row["card"]["status"] = "insufficient_evidence"
    row["card"]["claims"] = row["card"]["claims"][:1]
    issues = check_case(row, source)
    assert "expected_personal_card_missing" in issues
    assert "expected_personal_product_basis_missing" in issues
    _, row = artifact()
    row["evidence"]["feature"]["name"] = "okved_code"
    assert "expected_personal_product_basis_missing" in check_case(row, source)


def test_contract_checks_catch_exact_text_with_wrong_evidence_kind_and_scope():
    source, row = artifact()
    row["evidence"]["fact"]["kind"] = "company_feature"
    row["evidence"]["feature"]["company_id"] = "LLM_DEMO_OTHER"
    row["card"]["next_steps"][0]["basis_claim_ids"] = ["c1"]
    issues = check_case(row, source)
    assert {"factual_claim_not_exact_evidence", "evidence_company_mismatch", "inference_missing_personal_basis",
            "action_missing_inference_basis_or_conditions"} <= set(issues)


def test_quotes_geography_and_negative_control_are_checked_independently():
    source, row = artifact()
    row["event"]["facts"][0]["source_span"]["start"] = -1
    row["event"]["facts"][0]["attributes"]["geography"] = "US"
    issues = check_case(row, source)
    assert {"news_quote_span_mismatch", "expected_fact_attributes_missing"} <= set(issues)
    source.update(minimum_facts=0, maximum_facts=0, expected_status="insufficient_evidence")
    assert {"unexpected_extracted_facts", "unexpected_card_status"} <= set(check_case(row, source))


def test_injected_instruction_is_rejected_even_when_it_is_an_exact_quote():
    source, row = artifact()
    source["forbidden_output"] = ["ключевую"]
    assert check_case(row, source) == []  # A news quotation can legitimately contain a marker.
    source["forbid_instruction_facts"] = True
    issues = check_case(row, source)
    assert "injection_instruction_in_extracted_fact" in issues
    assert "injection_marker_in_generated_output" not in issues  # Exact factual quote is not obedience.
    row["card"]["claims"][1]["text"] += " ключевую 999 рублей."
    assert "injection_marker_in_generated_output" in check_case(row, source)
    row["card"]["claims"].append(copy.deepcopy(row["card"]["claims"][1]))
    assert {"number_in_model_inference", "duplicate_claim_ids"} <= set(check_case(row, source))


def test_question_must_be_completed_and_keep_the_parent_context():
    source, row = artifact()
    source["question"] = "Какие условия необходимо уточнить?"
    assert "requested_question_not_completed" in check_case(row, source, True)
    row["answer"] = {**copy.deepcopy(row["card"]), "id": "answer", "insight_id": "insight", "question": source["question"]}
    row["answer_evidence"] = row["evidence"]
    assert check_case(row, source, True) == []
    row["answer"]["role"] = "ceo"
    row["answer"]["question"] = "Подменённый вопрос"
    assert {"question_context_changed", "question_text_changed"} <= set(check_case(row, source, True))


def test_malformed_saved_artifacts_report_controls_instead_of_crashing():
    source, row = artifact()
    row["card"]["claims"][0]["evidence_ids"] = [None]
    assert "unknown_evidence_id" in check_case(row, source)
    assert check_result({"claims": "invalid"}, row["event"], source, {}) == ["invalid_result_structure"]


def test_recheck_is_offline_preserves_execution_failures_and_exposes_partial_suite():
    source, row = artifact()
    cases = {"version": "test", "scope": "development", "companies": [COMPANY], "roles": ["cfo"], "sources": [source]}
    report = {"case_hash": hashed(cases), "case_manifest": cases, "planned_cases": 2, "filtered_case_count": 4,
              "full_case_count": 8, "results": [row]}
    result = recheck_report(report)
    assert result["rechecked_without_model_calls"]
    assert not result["all_selected_controls_passed"] and not result["full_suite_complete"]
    assert result["not_run_cases"] == 1 and result["limited_selection"] and result["source_filter_applied"]
    assert report["results"][0]["status"] == "passed_controls"  # Caller data stays immutable.
    report["results"][0].update(status="failed", error="card_validation_failed")
    result = recheck_report(report)
    assert result["execution_failures"] == 1 and result["results"][0]["status"] == "failed"
    report["results"][0]["error"] = "raw body private-secret"
    assert recheck_report(report)["results"][0]["error"] == "evaluation_internal_error"
    report["case_manifest"]["sources"][0]["text"] += " изменено"
    with pytest.raises(AnalysisError, match="evaluation_case_manifest_changed"):
        recheck_report(report)


@pytest.mark.parametrize("mutation", [lambda c: c.update(roles=[]), lambda c: c["companies"].append(COMPANY),
    lambda c: c["sources"][0].update(minimum_facts=True), lambda c: c["sources"][0].update(maximum_facts=0)])
def test_invalid_case_manifest_is_rejected_before_any_work(tmp_path, mutation):
    source, _ = artifact()
    cases = {"version": "test", "scope": "development", "companies": [COMPANY], "roles": ["cfo"], "sources": [source]}
    mutation(cases)
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    with pytest.raises(AnalysisError, match="invalid_evaluation_cases"):
        read_cases(path)


def configured_environment(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'evaluation.db'}")
    monkeypatch.setenv("DEMO_COMPANY_IDS", "original-allowlist")
    for key, value in {"LLM_BASE_URL": "https://gateway.example.invalid/v1", "LLM_API_KEY": "private-evaluation-secret",
        "LLM_EXTRACTION_MODEL": "test-model", "LLM_ANALYSIS_MODEL": "test-model", "LLM_PRICES_CONFIRMED": "1",
        "LLM_BUDGET_MODE": "token_prices", "BANK_INDEX_DIR": str(tmp_path / "missing"),
        "LLM_MODEL_CATALOG_JSON": json.dumps([{"id": "test-model", "case_model": "qwen3.6-35b-a3b",
            "input_rub_per_million": "1", "output_rub_per_million": "2", "input_limit": 32768, "output_limit": 2500}])}.items():
        monkeypatch.setenv(key, value)
    return Settings.from_env()


def test_failed_worker_report_keeps_job_run_and_cost_diagnostics_without_raw_errors(tmp_path, monkeypatch):
    settings = configured_environment(monkeypatch, tmp_path)

    class FailingRuntime:
        def __init__(self, factory, settings):
            self.factory = factory

        def process(self, task, job, heartbeat):
            run_id = "run_" + hashed(job.id)[:32]
            with self.factory.begin() as db:
                db.add(AnalysisRun(id=run_id, company_id=job.company_id, role=task.role, as_of=task.as_of,
                    status="failed", steps=[{"action": "search_bank_materials", "raw_response": settings.api_key}],
                    versions={}, verification={"status": "failed", "issues": [settings.api_key]}, created_at=1))
                db.add(LLMCall(id="failed-call", run_id=run_id, day="2026-10-20", model="test-model",
                    purpose="extraction", request_hash="test", status="failed", reserved_micros=1234,
                    cost_micros=None, result={"secret": settings.api_key}, error=settings.api_key))
            raise RuntimeError("raw provider body " + settings.api_key)

    result = evaluate(settings, ROOT / "backend/fixtures/llm_cases.json", ROOT / "backend/fixtures/llm_companies.csv",
                      2, "data/processed/failure.json", runtime_factory=FailingRuntime)
    report_text = Path(result["report"]).read_text()
    report = json.loads(report_text)
    row = report["results"][0]
    assert result["execution_failures"] == 1 and result["not_run_cases"] == 1
    assert row["error"] == "analysis_failed" and row["jobs"]["analysis"]["status"] == "failed"
    assert row["attempt_runs"]["analysis"]["status"] == "failed"
    assert row["usage"]["unsettled_calls"] == 1 and row["usage"]["unsettled_reservation_rub"] == "0.001234"
    assert settings.api_key not in report_text and "raw provider body" not in report_text and "raw_response" not in report_text
    assert stat.S_IMODE(Path(result["report"]).stat().st_mode) == 0o600
    assert recheck_report(result["report"])["execution_failures"] == 1
    import os
    assert os.environ["DEMO_COMPANY_IDS"] == "original-allowlist"


def test_synthetic_database_guard_persists_incomplete_setup_report(tmp_path, monkeypatch):
    settings = configured_environment(monkeypatch, tmp_path)
    engine, factory = make_session_factory()
    Base.metadata.create_all(engine)
    with factory.begin() as db:
        db.add(Company(id="REAL_CUSTOMER", profile_version="test", products={}, countries=[], quality_flags=[], source_rows=[]))
    engine.dispose()
    with pytest.raises(AnalysisError, match="evaluation_requires_dedicated_synthetic_database"):
        evaluate(settings, ROOT / "backend/fixtures/llm_cases.json", ROOT / "backend/fixtures/llm_companies.csv",
                 2, "data/processed/guard.json", runtime_factory=lambda *args: pytest.fail("must not start runtime"))
    report = json.loads(Path("data/processed/guard.json").read_text())
    assert report["setup_error"] == "evaluation_requires_dedicated_synthetic_database"
    assert report["not_run_cases"] == 2 and not report["all_selected_controls_passed"]
    assert report["results"] == []


def test_successful_small_selection_is_not_reported_as_full_suite():
    _, row = artifact()
    result = summarize_report({"results": [row], "planned_cases": 1, "filtered_case_count": 24, "full_case_count": 24})
    assert result["all_selected_controls_passed"] and result["selected_suite_complete"]
    assert not result["full_suite_complete"] and result["limited_selection"]
    assert result["personal_cards"] == {"required_in_attempted_cases": 1, "ready": 1, "missing": 0}


@pytest.mark.parametrize("valid_report", [True, False], ids=["valid-v2-report", "invalid-report"])
def test_recheck_cli_without_credentials_never_initializes_database(tmp_path, valid_report):
    source, row = artifact()
    cases = {"version": "test", "scope": "development", "companies": [COMPANY], "roles": ["cfo"], "sources": [source]}
    report = {"report_version": "llm-development-report-v2", "case_hash": hashed(cases), "case_manifest": cases,
              "planned_cases": 1, "filtered_case_count": 1, "full_case_count": 1, "results": [row]}
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(report if valid_report else {"unexpected": "private malformed report detail"}))
    db_path = tmp_path / "must-not-be-created.db"
    # An isolated process catches import-time API creation, which in-process
    # monkeypatches miss once app.api has already been imported by other tests.
    env = {key: value for key, value in os.environ.items() if not key.startswith("LLM_")}
    env.update(DATABASE_URL=f"sqlite:///{db_path}",
               PYTHONPATH=os.pathsep.join(str(ROOT / path) for path in ("backend", "backend/src", "knowledge/src")),
               LLM_MODEL_CATALOG_JSON="invalid-settings-must-not-be-read-by-offline-command")
    result = subprocess.run([sys.executable, "-m", "app.intelligence.cli", "recheck", "--report", str(report_path)],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=15, check=False)
    payload = json.loads(result.stdout)
    if valid_report:
        assert result.returncode == 0, result.stderr
        assert payload["rechecked_without_model_calls"] and payload["all_selected_controls_passed"]
        assert payload["completed_cases"] == 1
    else:
        assert result.returncode == 2
        assert payload == {"status": "failed", "error": "invalid_evaluation_report"}
        assert "private malformed report detail" not in result.stdout + result.stderr
    assert "Traceback" not in result.stderr
    assert not db_path.exists(), "Offline imports must not create a database or API tables"
