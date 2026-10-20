"""No paid models: scripted HTTP gateway, real API/DB/workers and negative controls."""
import json
import time
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api import create_app
from app.core import Base, Company, PRODUCT_COLUMNS, import_companies, make_session_factory
from app.news import Event, Job, SourceDocument
from app.worker import process_next_job
from app.intelligence.evidence import verify_draft
from app.intelligence.extraction import extract
from app.intelligence.provider import Calls, Provider, reserve
from app.intelligence.schemas import CardDraft, Decision
from app.intelligence.service import Runtime
from app.intelligence.settings import AnalysisError, Model, Settings
from app.intelligence.storage import AnalysisRun, BudgetDay, Extraction, Insight, LLMCall
from app.intelligence.worker import process_next


CATALOG = [{"id": "test-organizer-model", "case_model": "qwen3.6-35b-a3b", "input_rub_per_million": "1",
            "output_rub_per_million": "2", "input_limit": 32768, "output_limit": 2500}]
BODY = "Банк России сохранил ключевую ставку. Условия кредитов и депозитов отдельных компаний не раскрыты."


@pytest.fixture
def store(tmp_path, monkeypatch):
    engine, factory = make_session_factory(f"sqlite:///{tmp_path / 'intelligence.db'}")
    Base.metadata.create_all(engine)
    import_companies(Path(__file__).resolve().parents[1] / "fixtures" / "companies.csv", factory)
    monkeypatch.setenv("DEMO_COMPANY_IDS", "DEMO_A,DEMO_B")
    monkeypatch.setenv("LLM_BASE_URL", "https://gateway.example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-secret-never-expose")
    monkeypatch.setenv("LLM_EXTRACTION_MODEL", "test-organizer-model")
    monkeypatch.setenv("LLM_ANALYSIS_MODEL", "test-organizer-model")
    monkeypatch.setenv("LLM_MODEL_CATALOG_JSON", json.dumps(CATALOG))
    monkeypatch.setenv("LLM_PRICES_CONFIRMED", "1")
    monkeypatch.setenv("BANK_INDEX_DIR", str(tmp_path / "missing-bank-index"))
    with TestClient(create_app(factory, engine)) as client:
        headers = {cid: {"X-Demo-Session": client.post("/api/v1/demo-session", json={"company_id": cid, "role": "cfo"}).json()["token"]}
                   for cid in ("DEMO_A", "DEMO_B")}
        yield client, factory, headers, Settings.from_env()
    engine.dispose()


class ScriptedGateway:
    """Fictional test responses derived from supplied evidence, never production fallback."""
    def __init__(self, *, bad_quote=False, bad_schema=False, reject_review=False, extra_search=False,
                 fabricated_number=False, timeout=False, missing_usage=False, empty_facts=False,
                 reject_semantics=False, repair_semantics=False, always_search=False):
        self.options = locals().copy()
        self.seen = []
        self.decisions = 0
        self.reviews = 0

    def __call__(self, request):
        body = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        assert body["stream"] is False
        assert "tools" not in body and "response_format" not in body
        schema = json.loads(body["messages"][0]["content"].split("schema:\n", 1)[1])
        data = json.loads(body["messages"][1]["content"])["untrusted_data"]
        title = schema["title"]
        self.seen.append((title, data))
        if self.options["timeout"]:
            raise httpx.ReadTimeout("secret text must never reach API", request=request)
        if title == "ExtractionDraft":
            quote = "invented quote absent from source" if self.options["bad_quote"] else data["text"].split(". ")[0] + "."
            result = {"facts": [] if self.options["empty_facts"] else [{"quote": quote,
                "attributes": {"topics": ["policy_rate"], "geography": "RU", "kind": "rate_decision"}}]}
        elif title == "ExtractionReview":
            result = {"accepted_indexes": [] if self.options["reject_review"] else list(range(len(data["candidates"]))), "issues": []}
        elif title in {"Decision", "FinalDecision"}:
            self.decisions += 1
            if title == "Decision" and ((self.options["extra_search"] and self.decisions == 1) or self.options["always_search"]):
                result = {"decision": {"action": "search_bank_materials", "query": f"Более свежий прогноз ключевой ставки, запрос {self.decisions}", "metric": "policy_rate", "horizon": "2026-12-31"}}
            else:
                news = next(e for e in data["evidence"] if e["kind"] == "event_fact")
                company = next(e for e in data["evidence"] if e["kind"] == "company_feature")
                claims = [{"id": "c1", "kind": "news_fact", "text": news["text"], "evidence_ids": [news["id"]], "conditions": []},
                          {"id": "c2", "kind": "company_fact", "text": company["text"], "evidence_ids": [company["id"]], "conditions": []},
                          {"id": "c3", "kind": "model_inference", "text": "Условия финансирования компании требуют проверки по договору.",
                           "evidence_ids": [news["id"], company["id"]], "conditions": ["Суммы и условия договоров неизвестны"]}]
                bank_claim = next((e for e in data["evidence"] if e["kind"] == "bank_claim" and e.get("position_role") == "current_in_document"), None)
                if bank_claim:
                    claims.append({"id": "c4", "kind": "bank_view", "text": bank_claim["text"],
                                   "evidence_ids": [bank_claim["id"]], "conditions": []})
                if self.options["fabricated_number"]:
                    claims[2]["text"] = "Компания потеряет 999999 рублей."
                result = {"decision": {"action": "finish", "draft": {"claims": claims, "risks": ["c3"], "opportunities": [],
                    "next_steps": [{"text": "Уточнить применимые условия договора", "basis_claim_ids": ["c2", "c3"],
                                    "conditions": ["Если компания планирует пересмотр условий"]}], "missing_data": ["Условия договора"]}}}
        elif title == "SemanticReview":
            self.reviews += 1
            result = {"supported_claim_ids": [c["id"] for c in data["draft"]["claims"]],
                      "supported_action_indexes": list(range(len(data["draft"]["next_steps"]))), "issues": []}
            if self.options["reject_semantics"] or (self.options["repair_semantics"] and self.reviews == 1):
                result["supported_claim_ids"] = []
                result["issues"] = ["Unsupported interpretation"]
        else:
            raise AssertionError(title)
        if self.options["bad_schema"]:
            result["run_shell"] = "must not execute"
        payload = {"choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}, "finish_reason": "stop"}]}
        if not self.options["missing_usage"]:
            payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 80}
        return httpx.Response(200, json=payload)


def runtime(factory, settings, gateway=None):
    gateway = gateway or ScriptedGateway()
    client = httpx.Client(transport=httpx.MockTransport(gateway))
    return Runtime(factory, settings, Provider(factory, settings, client)), gateway


def stored_event(client, factory, headers, *, text=BODY, title="Тестовое решение Банка России"):
    result = client.post("/api/v1/publications", headers={**headers, "Idempotency-Key": "publish-test-001"},
                         json={"title": title, "text": text})
    assert result.status_code == 202, result.text
    assert process_next_job(factory)
    job = client.get("/api/v1/jobs/" + result.json()["id"], headers=headers).json()
    assert job["status"] == "succeeded"
    return job["result_id"]


def queued_insight(client, headers, event_id, key="analysis-test-001", **extra):
    return client.post("/api/v1/companies/DEMO_A/insights", headers={**headers, "Idempotency-Key": key},
                       json={"event_id": event_id, **extra})


def test_full_api_extraction_agent_extra_search_evidence_and_budget(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    queued = queued_insight(client, headers["DEMO_A"], event_id)
    assert queued.status_code == 202, queued.text
    job_id = queued.json()["id"]
    # The raw ingestion worker must never steal a paid analysis task.
    assert not process_next_job(factory)
    engine, gateway = runtime(factory, settings, ScriptedGateway(extra_search=True))
    assert process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "succeeded", job
    card = client.get("/api/v1/insights/" + job["result_id"], headers=headers["DEMO_A"]).json()
    assert card["status"] == "ready_with_conditions"
    assert card["bank_view"]["status"] == "not_found"
    assert card["verification"]["status"] == "passed_with_conditions"
    assert card["event_version"] == "2"
    record = client.get("/api/v1/events/" + event_id, headers=headers["DEMO_A"]).json()
    assert len(record["facts"]) == 1
    fact = record["facts"][0]
    assert BODY[fact["source_span"]["start"]:fact["source_span"]["end"]] == fact["text"]
    assert fact["verification_method"] == "exact_quote_and_separate_llm_entailment"
    for claim in card["claims"]:
        for evidence_id in claim["evidence_ids"]:
            evidence = client.get(f"/api/v1/evidence/{evidence_id}", params={"insight_id": card["id"]}, headers=headers["DEMO_A"])
            assert evidence.status_code == 200
            assert client.get(f"/api/v1/evidence/{evidence_id}", params={"insight_id": card["id"]}, headers=headers["DEMO_B"]).status_code == 404
    trace = client.get("/api/v1/runs/" + card["run_id"], headers=headers["DEMO_A"]).json()
    assert any(s.get("origin") != "initial_context" and s["action"] == "search_bank_materials" for s in trace["steps"])
    assert len(trace["llm_calls"]) == 5
    assert "test-secret-never-expose" not in json.dumps(trace)
    assert client.get("/api/v1/runs/" + card["run_id"], headers=headers["DEMO_B"]).status_code == 404
    budget = client.get("/api/v1/llm/status", headers=headers["DEMO_A"]).json()
    assert Decimal(budget["spent_rub"]) > 0 and Decimal(budget["reserved_rub"]) == 0


def test_question_keeps_original_profile_role_date_and_evidence(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings)
    process_next(factory, engine)
    card_id = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()["result_id"]
    first = client.get("/api/v1/insights/" + card_id, headers=headers["DEMO_A"]).json()
    with factory.begin() as db:
        db.get(Company, "DEMO_A").profile_version = "new-profile-version"
    result = client.post(f"/api/v1/insights/{card_id}/questions", headers={**headers["DEMO_A"], "Idempotency-Key": "question-test-001"},
                         json={"question": "Какие данные стоит уточнить перед выводом?"})
    assert result.status_code == 202, result.text
    assert process_next(factory, engine)
    answer_job = client.get("/api/v1/jobs/" + result.json()["id"], headers=headers["DEMO_A"]).json()
    assert answer_job["status"] == "succeeded", answer_job
    answer = client.get("/api/v1/answers/" + answer_job["result_id"], headers=headers["DEMO_A"]).json()
    assert answer["as_of"] == first["as_of"]
    assert answer["profile_version"] == first["profile_version"]
    assert sum(title == "ExtractionDraft" for title, _ in gateway.seen) == 1
    assert client.get("/api/v1/answers/" + answer["id"], headers=headers["DEMO_B"]).status_code == 404


@pytest.mark.parametrize("option", ["bad_quote", "reject_review", "empty_facts"])
def test_unverified_facts_produce_no_personal_inference(store, option):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job = queued_insight(client, headers["DEMO_A"], event_id).json()
    engine, gateway = runtime(factory, settings, ScriptedGateway(**{option: True}))
    process_next(factory, engine)
    result = client.get("/api/v1/jobs/" + job["id"], headers=headers["DEMO_A"]).json()
    assert result["status"] == "succeeded", result
    card = client.get("/api/v1/insights/" + result["result_id"], headers=headers["DEMO_A"]).json()
    assert card["status"] == "insufficient_evidence"
    assert not any(c["kind"] == "model_inference" for c in card["claims"])
    assert not any(title == "Decision" for title, _ in gateway.seen)


@pytest.mark.parametrize("option,error", [("timeout", "llm_timeout_outcome_unknown"), ("missing_usage", "llm_usage_missing"),
                                          ("bad_schema", "llm_schema_invalid"), ("fabricated_number", "card_validation_failed")])
def test_failed_model_calls_do_not_publish_or_retry_paid_requests(store, option, error):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings, ScriptedGateway(**{option: True}))
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "failed" and job["error"] == error, job
    assert not process_next(factory, engine)
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Insight)) == 0
        budget = db.scalar(select(BudgetDay))
        if option in {"timeout", "missing_usage"}:
            assert budget.reserved_micros > 0
        assert len(gateway.seen) <= settings.max_calls


def test_budget_limit_blocks_before_http(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, replace(settings, daily_limit_micros=1))
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "failed" and job["error"] == "budget_exceeded"
    assert gateway.seen == []


def test_idempotency_company_role_and_unconfigured_boundaries(store, monkeypatch):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    assert client.post("/api/v1/companies/DEMO_A/insights", json={"event_id": event_id}).status_code == 401
    assert queued_insight(client, headers["DEMO_B"], event_id).status_code == 403
    first = queued_insight(client, headers["DEMO_A"], event_id)
    assert queued_insight(client, headers["DEMO_A"], event_id).json() == first.json()
    assert queued_insight(client, headers["DEMO_A"], event_id, as_of="2026-09-01T00:00:00Z").status_code == 409
    assert queued_insight(client, headers["DEMO_A"], event_id, key="nozone-time-01", as_of="2026-09-01T00:00:00").status_code == 422
    monkeypatch.delenv("LLM_API_KEY")
    assert queued_insight(client, headers["DEMO_A"], event_id, key="unconfigured-01").status_code == 503
    assert not client.get("/api/v1/llm/status", headers=headers["DEMO_A"]).json()["configured"]


def test_cache_reuses_verified_insight_and_extraction(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    when = datetime.now(timezone.utc).isoformat()
    engine, gateway = runtime(factory, settings)
    results = []
    for key in ("cached-analysis-001", "cached-analysis-002"):
        job_id = queued_insight(client, headers["DEMO_A"], event_id, key=key, as_of=when).json()["id"]
        process_next(factory, engine)
        job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
        assert job["status"] == "succeeded", job
        results.append(job["result_id"])
    assert len(gateway.seen) == 4
    assert results[0] == results[1]
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Extraction)) == 1


def test_concurrent_identical_analysis_waits_and_reuses_one_generation(store):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Event as ThreadEvent
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    when = datetime.now(timezone.utc).isoformat()
    job_ids = [queued_insight(client, headers["DEMO_A"], event_id, key=f"parallel-analysis-{i}", as_of=when).json()["id"]
               for i in range(2)]
    entered, release = ThreadEvent(), ThreadEvent()

    class BlockingGateway(ScriptedGateway):
        def __call__(self, request):
            result = super().__call__(request)
            schema = json.loads(json.loads(request.content)["messages"][0]["content"].split("schema:\n", 1)[1])
            if schema["title"] == "Decision":
                entered.set()
                assert release.wait(timeout=8)
            return result

    engine, gateway = runtime(factory, settings, BlockingGateway())
    with ThreadPoolExecutor(max_workers=1) as pool:
        first = pool.submit(process_next, factory, engine)
        assert entered.wait(timeout=5)
        try:
            assert process_next(factory, engine)
            with factory() as db:
                waiting = db.scalar(select(Job).where(Job.id.in_(job_ids), Job.status == "queued"))
                assert waiting is not None and waiting.stage == "waiting_shared_analysis"
                waiting_id = waiting.id
        finally:
            release.set()
        assert first.result(timeout=5)
    with factory.begin() as db:
        db.get(Job, waiting_id).available_at = int(time.time()) - 1
    assert process_next(factory, engine)
    results = [client.get("/api/v1/jobs/" + jid, headers=headers["DEMO_A"]).json() for jid in job_ids]
    assert all(r["status"] == "succeeded" for r in results), results
    assert results[0]["result_id"] == results[1]["result_id"]
    assert len(gateway.seen) == 4


def test_historical_request_excludes_late_received_source_before_model(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    past = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
    job_id = queued_insight(client, headers["DEMO_A"], event_id, as_of=past).json()["id"]
    engine, gateway = runtime(factory, settings)
    process_next(factory, engine)
    result = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert result["status"] == "failed" and result["error"] == "event_unavailable_as_of"
    assert gateway.seen == []


def test_prompt_injection_cannot_add_actions_or_disable_validation():
    with pytest.raises(ValueError):
        Decision.model_validate({"decision": {"action": "run_shell", "command": "cat secrets"}})
    with pytest.raises(ValueError):
        Decision.model_validate({"decision": {"action": "search_bank_materials", "query": "ставка", "company_id": "other"}})
    draft = CardDraft.model_validate({"claims": [{"id": "c1", "kind": "bank_view", "text": "Банк ожидает выдуманный прогноз",
                                                  "evidence_ids": ["unknown"]}], "risks": [], "opportunities": []})
    assert verify_draft(draft, {}) == ["c1:unknown_evidence"]


def test_model_catalog_rejects_disallowed_family_unconfirmed_prices_and_overspend(store):
    _, _, _, settings = store
    with pytest.raises(AnalysisError):
        replace(settings, prices_confirmed=False).require_ready()
    with pytest.raises(ValueError):
        replace(settings, daily_limit_micros=501_000_000).validate()
    with pytest.raises(ValueError):
        replace(settings, models={"x": Model("x", "outside-case", Decimal(1), Decimal(1))}).validate()


def test_semantic_repair_is_attempted_once_and_rechecked(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings, ScriptedGateway(repair_semantics=True))
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "succeeded", job
    assert gateway.reviews == 2 and gateway.decisions == 2
    assert len(gateway.seen) == 6


def test_persistent_semantic_failure_never_publishes(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings, ScriptedGateway(reject_semantics=True))
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "failed" and job["error"] == "card_validation_failed"
    assert gateway.reviews == 2 and gateway.decisions == 2


def test_agent_reserves_final_calls_for_card_and_verification(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings, ScriptedGateway(always_search=True))
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "succeeded", job
    assert len(gateway.seen) == settings.max_calls
    assert gateway.seen[-2][0] == "FinalDecision"
    assert gateway.seen[-1][0] == "SemanticReview"


def test_partial_extraction_is_disclosed_and_keeps_exact_offsets(store):
    client, factory, headers, settings = store
    body = BODY + " " + "Подробности публикации. " * 600
    event_id = stored_event(client, factory, headers["DEMO_A"], text=body)
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, _ = runtime(factory, settings)
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "succeeded", job
    card = client.get("/api/v1/insights/" + job["result_id"], headers=headers["DEMO_A"]).json()
    assert "partial_extraction_bounded_source_prefix" in card["limitations"]
    assert card["extraction_coverage"]["complete"] is False
    assert card["extraction_coverage"]["end"] < len(body)


def test_shared_atomic_budget_and_run_call_cap(store):
    from concurrent.futures import ThreadPoolExecutor
    _, factory, _, settings = store
    limited = replace(settings, daily_limit_micros=80_000, max_calls=4)
    model = limited.models[limited.analysis_model]

    def attempt(index):
        try:
            reserve(factory, limited, f"call-{index}", f"run-{index}", model, "test", str(index))
            return True
        except AnalysisError as exc:
            assert exc.code == "budget_exceeded"
            return False
    with ThreadPoolExecutor(max_workers=6) as pool:
        accepted = sum(pool.map(attempt, range(6)))
    assert accepted == 2
    with factory() as db:
        budget = db.scalar(select(BudgetDay))
        assert budget.spent_micros + budget.reserved_micros <= 80_000


def test_fresh_model_is_not_called_twice_for_an_indeterminate_call(store):
    _, factory, _, settings = store
    gateway = ScriptedGateway(timeout=True)
    provider = runtime(factory, settings, gateway)[0].provider
    for index in range(2):
        with pytest.raises(AnalysisError) as error:
            provider.generate("run-retry", "same-purpose", settings.analysis_model, "system", {"news": "test"}, {"title": "Decision"})
        assert error.value.code == ("llm_timeout_outcome_unknown" if index == 0 else "previous_call_outcome_unknown")
    assert len(gateway.seen) == 1


def test_restarted_step_with_changed_prompt_does_not_resend_paid_call(store):
    _, factory, _, settings = store
    gateway = ScriptedGateway(timeout=True)
    provider = runtime(factory, settings, gateway)[0].provider
    for message in ("first", "changed"):
        with pytest.raises(AnalysisError) as error:
            provider.generate("same-run", "same-step", settings.analysis_model, "system", {"news": message}, {"title": "Decision"})
        assert error.value.code == ("llm_timeout_outcome_unknown" if message == "first" else "resumed_call_context_changed")
    assert len(gateway.seen) == 1


def test_old_bank_forecast_cannot_override_newer_matching_forecast():
    old = {"id": "old", "kind": "bank_claim", "text": "Банк ожидал прежнюю ставку.", "metric": "policy_rate",
           "forecast_horizon": "2026-12-31", "geography": "RU", "scenario": "base", "attributed_to": "alfa_bank",
           "position_role": "current_in_document", "published_at": "2026-07-31"}
    newer = {**old, "id": "new", "published_at": "2026-09-11", "text": "Банк пересмотрел ожидаемую ставку."}
    draft = CardDraft.model_validate({"claims": [{"id": "c1", "kind": "bank_view", "text": old["text"], "evidence_ids": ["old"]}]})
    assert "c1:bank_claim_superseded_in_context" in verify_draft(draft, {"old": old, "new": newer})


def test_llm_job_history_feed_and_role_isolation(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, _ = runtime(factory, settings)
    process_next(factory, engine)
    jobs = client.get("/api/v1/jobs?operation=insight_analysis", headers=headers["DEMO_A"]).json()["items"]
    assert [j["id"] for j in jobs] == [job_id]
    assert client.get("/api/v1/jobs?operation=insight_analysis", headers=headers["DEMO_B"]).json()["items"] == []
    assert client.get("/api/v1/jobs?operation=invalid", headers=headers["DEMO_A"]).status_code == 422
    feed = client.get("/api/v1/companies/DEMO_A/feed", headers=headers["DEMO_A"])
    assert feed.status_code == 200, feed.text
    item = next(i for i in feed.json()["items"] if i["event_id"] == event_id)
    assert item["insight"]["status"] == "ready_with_conditions"
    ceo = {"X-Demo-Session": client.post("/api/v1/demo-session", json={"company_id": "DEMO_A", "role": "ceo"}).json()["token"]}
    assert client.get("/api/v1/insights/" + item["insight"]["id"], headers=ceo).status_code == 404


def test_historical_feed_filters_time_before_bounded_candidate_pool(store):
    client, factory, headers, _ = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    now = datetime.now(timezone.utc)
    with factory.begin() as db:
        old = db.get(Event, event_id)
        source = db.get(SourceDocument, old.source_id)
        source.received_at = (now - timedelta(days=2)).isoformat()
        for i in range(105):
            source_id = f"src_future_{i}"
            db.add(SourceDocument(id=source_id, company_id="DEMO_A", title=f"Later source {i}", body=BODY,
                                  received_at=now.isoformat(), content_hash=f"unique_future_hash_{i}"))
            db.add(Event(id=f"event_future_{i}", company_id="DEMO_A", source_id=source_id, title=f"Later source {i}"))
    response = client.get("/api/v1/companies/DEMO_A/feed", params={"as_of": (now - timedelta(days=1)).isoformat()}, headers=headers["DEMO_A"])
    assert response.status_code == 200, response.text
    assert [item["event_id"] for item in response.json()["items"]] == [event_id]


def test_call_limit_persists_between_worker_attempts(store):
    _, factory, _, settings = store
    limited = replace(settings, max_calls=4)
    model = limited.models[limited.analysis_model]
    for index in range(4):
        reserve(factory, limited, f"persisted-{index}", "persistent-run", model, "test", str(index))
    with pytest.raises(AnalysisError, match="agent_call_limit"):
        reserve(factory, limited, "persisted-fifth", "persistent-run", model, "test", "fifth")


@pytest.mark.skipif(not (Path(__file__).resolve().parents[2] / "data/processed/bank-index-v1/manifest.json").exists(),
                    reason="Optional local integration requires the prepared bank index and pinned E5 assets")
def test_real_bank_index_and_local_e5_with_scripted_generation(store):
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    client, factory, headers, settings = store
    root = Path(__file__).resolve().parents[2]
    settings = replace(settings, bank_index=str(root / "data/processed/bank-index-v1"),
                       model_dir=str(root / "data/models/e5-base"), model_manifest=str(root / "knowledge/embedding-model.json"))
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings)
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "succeeded", job
    card = client.get("/api/v1/insights/" + job["result_id"], headers=headers["DEMO_A"]).json()
    assert card["bank_view"]["status"] == "found"
    assert any(c["kind"] == "bank_view" for c in card["claims"])
    with factory() as db:
        saved = db.get(Insight, card["id"])
        bank_claims = [e for e in saved.evidence.values() if e["kind"] == "bank_claim"]
        assert any(e["metric"] == "policy_rate" and e["forecast_horizon"] == "2026-12-31" and e["value"] == "14" for e in bank_claims)
    assert len(gateway.seen) == 4
    assert engine.encoder is not None


def test_actual_loopback_http_gateway_and_usage_accounting(store):
    from http.server import BaseHTTPRequestHandler
    from socketserver import ThreadingTCPServer
    from threading import Thread
    from app.intelligence.schemas import ExtractionDraft
    _, factory, _, settings = store
    gateway = ScriptedGateway()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            request = httpx.Request("POST", "http://127.0.0.1" + self.path,
                                    content=self.rfile.read(int(self.headers["Content-Length"])))
            response = gateway(request)
            body = response.content
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with ThreadingTCPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            configured = replace(settings, base_url=f"http://127.0.0.1:{server.server_address[1]}/v1")
            with httpx.Client(trust_env=False, timeout=3) as client:
                result = Provider(factory, configured, client).generate("wire-run", "wire-test", settings.extraction_model,
                    "Extract literal quotes", {"text": BODY, "title": "Тестовая публикация"}, ExtractionDraft.model_json_schema())
            assert ExtractionDraft.model_validate(result).facts[0].quote in BODY
            with factory() as db:
                call = db.scalar(select(LLMCall).where(LLMCall.run_id == "wire-run"))
                assert call.status == "completed" and call.cost_micros > 0
        finally:
            server.shutdown()
            thread.join(timeout=3)
