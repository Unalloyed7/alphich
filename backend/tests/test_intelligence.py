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
from app.intelligence.storage import AnalysisRun, AnalysisTask, BudgetDay, Extraction, Insight, LLMCall
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


def test_gateway_status_is_session_scoped_read_only_and_sanitized(store, monkeypatch):
    client, factory, headers, _ = store
    from app.intelligence import routes
    seen = []
    def inspect(settings):
        seen.append(settings.base_url)
        return {"status": "reachable", "selected_models_available": True}
    monkeypatch.setattr(routes, "inspect_gateway", inspect)
    assert client.get('/api/v1/llm/gateway-status').status_code == 401
    response = client.get('/api/v1/llm/gateway-status', headers=headers['DEMO_A'])
    assert response.status_code == 200 and response.json()['selected_models_available'] is True
    assert len(seen) == 1 and 'test-secret' not in response.text
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0
        assert db.scalar(select(func.count()).select_from(BudgetDay)) == 0


def test_gateway_status_returns_safe_upstream_error(store, monkeypatch):
    client, _, headers, _ = store
    from app.intelligence import routes
    def inspect(_):
        raise AnalysisError('gateway_models_http_503')
    monkeypatch.setattr(routes, 'inspect_gateway', inspect)
    response = client.get('/api/v1/llm/gateway-status', headers=headers['DEMO_A'])
    assert response.status_code == 503 and response.json()['detail'] == 'gateway_models_http_503'


class ScriptedGateway:
    """Fictional test responses derived from supplied evidence, never production fallback."""
    def __init__(self, *, bad_quote=False, bad_schema=False, reject_review=False, extra_search=False,
                 fabricated_number=False, timeout=False, missing_usage=False, empty_facts=False,
                 reject_semantics=False, repair_semantics=False, always_search=False, bad_contract_stability=False,
                 business_issue=None, persistent_business_issue=False, relevance_grade=4):
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
                "attributes": {"topics": ["policy_rate"], "geography": "RU", "kind": "rate_decision",
                    "geographic_scope": "national", "market_scope": "sector_wide",
                    "impact_scale": "material",
                    "business_effect": "financing", "event_status": "reported"}}]}
        elif title == "ExtractionReview":
            result = {"accepted_indexes": [] if self.options["reject_review"] else list(range(len(data["candidates"]))), "issues": []}
        elif title == "RelevanceJudgment":
            # A fixed test verdict exercises orchestration, never grading quality.
            result = {"grade": self.options["relevance_grade"] if data["news_facts"] else 0,
                      "reason": "Проверенный факт о ставке связан с подтверждённой деятельностью компании; условия договоров неизвестны.",
                      "news_fact_ids": [fact["id"] for fact in data["news_facts"][:1]],
                      "company_feature_ids": [feature["id"] for feature in data["company_features"][:1]],
                      "missing_data": [], "required_action": None}
        elif title in {"ResearchDecision", "FinalResearchDecision"}:
            result = {"decision": {"action": "handoff", "bank_claim_ids": data["eligible_bank_claim_ids"],
                                    "gaps": [] if data["eligible_bank_claim_ids"] else ["no_applicable_evidence"]}}
        elif title in {"Decision", "FinalDecision"}:
            self.decisions += 1
            if title == "Decision" and ((self.options["extra_search"] and self.decisions == 1) or self.options["always_search"]):
                result = {"decision": {"action": "search_bank_materials", "query": f"Более свежий прогноз ключевой ставки, запрос {self.decisions}", "metric": "policy_rate", "horizon": "2026-12-31"}}
            else:
                news = next(e for e in data["evidence"] if e["kind"] == "event_fact")
                company = next(e for e in data["evidence"] if e["kind"] == "company_feature" and "Основная деятельность" in e["text"])
                facts = [news["id"], company["id"]]
                inferences = [{"id": "i1", "text": "Если заказчики строительной компании финансируют проекты кредитами, сохранение дорогого финансирования может откладывать запуск проектов и поступление авансов подрядчику.",
                               "evidence_ids": [news["id"], company["id"]], "conditions": ["Проекты заказчиков зависят от кредитного финансирования"]}]
                bank_claim = next((e for e in data["evidence"] if e["kind"] == "bank_claim" and e.get("position_role") == "current_in_document"), None)
                if bank_claim:
                    facts.append(bank_claim["id"])
                if self.options["fabricated_number"]:
                    inferences[0]["text"] = "Компания потеряет 999999 рублей."
                if self.options["bad_contract_stability"] and self.decisions == 1:
                    inferences[0]["text"] = ("Если у компании есть действующий кредитный договор, условия финансирования "
                        "могут сохраняться в неизменном виде до конца года в базовом сценарии банка; "
                        "стоимость и структуру долга следует уточнить по договору.")
                if self.options["business_issue"] and (self.decisions == 1 or self.options["persistent_business_issue"]):
                    inferences[0]["text"] = {
                        "missing_business_mechanism": "Условия финансирования компании требуют проверки по договору.",
                        "unsupported_transmission": "Если ставка остаётся высокой, заказчики стремятся больше тратить и ускоряют запуск строительных проектов.",
                        "off_focus_inference": "Если у компании есть действующий кредит, его обслуживание может ограничивать денежный поток; нужно оценить график платежей.",
                    }[self.options["business_issue"]]
                result = {"decision": {"action": "finish", "draft": {"fact_evidence_ids": facts, "inferences": inferences, "risks": ["i1"], "opportunities": [],
                    "next_steps": [{"text": "Проверить подтверждённое финансирование и сроки запуска крупных заявок, чтобы оценить надёжность плана поступления авансов.", "basis_inference_ids": ["i1"],
                                    "conditions": ["Если в плане есть ещё не запущенные проекты"]}], "missing_data": ["Источники финансирования и сроки запуска проектов заказчиков"]}}}
        elif title in {"SemanticReview", "VerificationDecision", "FinalVerificationDecision"}:
            self.reviews += 1
            result = {"supported_claim_ids": [c["id"] for c in data["draft"]["claims"]],
                      "supported_action_indexes": list(range(len(data["draft"]["next_steps"]))), "issues": []}
            if self.options["reject_semantics"] or (self.options["repair_semantics"] and self.reviews == 1):
                result["supported_claim_ids"] = []
                result["issues"] = ["unsupported_claim"]
            if self.options["business_issue"] and (self.reviews == 1 or self.options["persistent_business_issue"]):
                result["supported_claim_ids"] = [c["id"] for c in data["draft"]["claims"] if c["kind"] != "model_inference"]
                result["issues"] = [self.options["business_issue"]]
            if title != "SemanticReview":
                result = {"decision": {"action": "verdict", **result}}
        else:
            raise AssertionError(title)
        if self.options["bad_schema"]:
            result["run_shell"] = "must not execute"
        payload = {"choices": [{"message": {"content": json.dumps(result, ensure_ascii=False)}, "finish_reason": "stop"}]}
        if not self.options["missing_usage"]:
            payload["usage"] = {"prompt_tokens": 100, "completion_tokens": 80}
        return httpx.Response(200, json=payload)


class FakeE5:
    """Deterministic unit vectors for protocol tests, not a semantic model."""
    dimension = 2
    space_id = "test-e5-normalized-v1"

    def count_tokens(self, text):
        return len(text.split())

    def encode_query(self, text):
        import numpy as np
        return np.asarray([1.0, 0.0], dtype=np.float32)

    def encode_passages(self, texts, batch_size=4):
        import numpy as np
        return np.tile(np.asarray([1.0, 0.0], dtype=np.float32), (len(texts), 1))


def runtime(factory, settings, gateway=None, *, fake_e5=True, client=None):
    gateway = gateway or ScriptedGateway()
    client = client or httpx.Client(transport=httpx.MockTransport(gateway))
    worker = Runtime(factory, settings, Provider(factory, settings, client))
    if fake_e5:
        worker.encoder = FakeE5()
    return worker, gateway


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


def test_fact_only_route_is_scoped_idempotent_and_does_not_generate_personal_card(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers['DEMO_A'])
    path = '/api/v1/events/' + event_id + '/extract-facts'
    assert client.post(path).status_code == 401
    assert client.post(path, headers={**headers['DEMO_B'], 'Idempotency-Key': 'facts-route-test-001'}).status_code == 404
    assert client.post(path, headers=headers['DEMO_A']).status_code == 400
    own = {**headers['DEMO_A'], 'Idempotency-Key': 'facts-route-test-001'}
    response = client.post(path, headers=own)
    assert response.status_code == 202, response.text
    assert response.json()['result_type'] == 'event'
    assert client.post(path, headers=own).json()['id'] == response.json()['id']
    engine, gateway = runtime(factory, settings)
    assert process_next(factory, engine)
    job = client.get('/api/v1/jobs/' + response.json()['id'], headers=headers['DEMO_A']).json()
    assert job['status'] == 'succeeded' and job['result_id'] == event_id
    assert [title for title, _ in gateway.seen] == ['ExtractionDraft', 'ExtractionReview', 'RelevanceJudgment']
    event = client.get('/api/v1/events/' + event_id, headers=headers['DEMO_A']).json()
    assert event['facts']
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Insight)) == 0
        assert db.scalar(select(func.count()).select_from(AnalysisTask)) == 1
        assert db.scalar(select(BudgetDay)).spent_micros > 0
    # A different operation may use the same idempotency key, and reuses facts.
    full = queued_insight(client, headers['DEMO_A'], event_id, key='facts-route-test-001').json()
    assert full['id'] != job['id']
    assert process_next(factory, engine)
    assert sum(title == 'ExtractionDraft' for title, _ in gateway.seen) == 1


def test_business_context_and_quality_are_scoped_and_free_to_read(store):
    client, factory, headers, settings = store
    path = '/api/v1/companies/DEMO_A/business-context'
    assert client.get(path).status_code == 401
    assert client.get(path, headers=headers['DEMO_B']).status_code == 403
    context = client.get(path, headers=headers['DEMO_A']).json()
    assert context['company_id'] == 'DEMO_A' and context['hypotheses']
    assert context['market_position']['market_share'] is None
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0
    event_id = stored_event(client, factory, headers['DEMO_A'])
    job = queued_insight(client, headers['DEMO_A'], event_id).json()
    engine, _ = runtime(factory, settings)
    assert process_next(factory, engine)
    result = client.get('/api/v1/jobs/' + job['id'], headers=headers['DEMO_A']).json()
    assert result['status'] == 'succeeded', result
    quality_path = '/api/v1/insights/' + result['result_id'] + '/quality'
    assert client.get(quality_path, headers=headers['DEMO_B']).status_code == 404
    quality = client.get(quality_path, headers=headers['DEMO_A']).json()
    assert quality['factual_quote_match']['ratio'] == 1
    assert quality['semantic_correctness'] is None
    assert quality['human_review']['status'] == 'not_recorded'


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
    assert len(trace["llm_calls"]) == 6
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
    assert result["result_type"] == "event" and result["result_id"] == event_id
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Insight)) == 0
    assert not any(title == "Decision" for title, _ in gateway.seen)


@pytest.mark.parametrize("option,error", [("timeout", "llm_timeout_outcome_unknown"), ("missing_usage", "llm_usage_missing"),
                                          ("bad_schema", "llm_schema_invalid"), ("fabricated_number", "llm_schema_invalid")])
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
            assert len(gateway.seen) == 1
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


def test_gateway_receipt_reserves_remaining_budget_and_accounts_exact_cost(store):
    _, factory, _, settings = store
    settings = replace(settings, prices_confirmed=False, budget_mode="gateway_receipt", receipt_rub_per_unit=Decimal("1"))
    seen = []
    def gateway(request):
        with factory() as db:
            budget = db.scalar(select(BudgetDay))
            assert budget.spent_micros + budget.reserved_micros == budget.limit_micros
        seen.append(request)
        return httpx.Response(200, headers={"x-litellm-response-cost": "0.1234567"}, json={
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        provider = Provider(factory, settings, client)
        for i in range(2):
            assert provider.generate("receipt-run", f"call-{i}", settings.analysis_model, "Return JSON", {}, {}) == {"ok": True}
    with factory() as db:
        budget = db.scalar(select(BudgetDay))
        assert budget.spent_micros == 246914
        assert budget.reserved_micros == 0
    assert len(seen) == 2


@pytest.mark.parametrize("receipt", [None, "NaN", "-1", "inf", "1e9999", "not-a-price"])
def test_missing_or_invalid_cost_receipt_keeps_full_reserve_and_blocks_next_call(store, receipt):
    _, factory, _, settings = store
    settings = replace(settings, prices_confirmed=False, budget_mode="gateway_receipt", receipt_rub_per_unit=Decimal("1"))
    seen = []
    def gateway(request):
        seen.append(request)
        return httpx.Response(200, headers={} if receipt is None else {"x-litellm-response-cost": receipt}, json={
            "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 5}})
    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        provider = Provider(factory, settings, client)
        with pytest.raises(AnalysisError, match="cost_unknown_reserve_retained"):
            provider.generate("receipt-run", "unknown", settings.analysis_model, "Return JSON", {}, {})
        with pytest.raises(AnalysisError, match="budget_exceeded"):
            provider.generate("receipt-run", "next", settings.analysis_model, "Return JSON", {}, {})
    with factory() as db:
        budget = db.scalar(select(BudgetDay))
        assert budget.spent_micros == 0 and budget.reserved_micros == settings.daily_limit_micros
    assert len(seen) == 1


def test_receipt_mode_allows_missing_tariffs_but_requires_explicit_currency(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "https://gateway.example.invalid/v1")
    monkeypatch.setenv("LLM_API_KEY", "test-secret")
    monkeypatch.setenv("LLM_MODEL_CATALOG_JSON", json.dumps([{k: v for k, v in CATALOG[0].items() if "rub_per" not in k}]))
    monkeypatch.setenv("LLM_EXTRACTION_MODEL", CATALOG[0]["id"])
    monkeypatch.setenv("LLM_ANALYSIS_MODEL", CATALOG[0]["id"])
    monkeypatch.setenv("LLM_PRICES_CONFIRMED", "0")
    monkeypatch.setenv("LLM_BUDGET_MODE", "gateway_receipt")
    monkeypatch.delenv("LLM_RECEIPT_RUB_PER_UNIT", raising=False)
    with pytest.raises(AnalysisError, match="llm_not_configured"):
        Settings.from_env().require_ready()
    monkeypatch.setenv("LLM_RECEIPT_RUB_PER_UNIT", "1")
    Settings.from_env().require_ready()


def test_read_only_gateway_status_redacts_key_and_excludes_unapproved_models(store):
    from app.intelligence.gateway import inspect_gateway
    _, _, _, settings = store
    paths = []
    def gateway(request):
        paths.append(request.url.path)
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": settings.analysis_model}, {"id": "deepseek-cost-test"}]})
        return httpx.Response(200, json={"key": settings.api_key, "info": {"key": settings.api_key,
            "spend": 1, "max_budget": 500, "budget_duration": "1d", "metadata": {"secret": settings.api_key}}})
    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        result = inspect_gateway(settings, client)
    assert result["allowed_models_available"] == [settings.analysis_model]
    assert settings.api_key not in json.dumps(result)
    assert paths == ["/v1/models", "/key/info"]


def test_evaluation_runs_api_workers_question_and_writes_local_report(store, tmp_path, monkeypatch):
    from app.intelligence.evaluation import evaluate
    _, _, _, settings = store
    root = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path / 'evaluation.db'}")
    cases = {"version": "test-v1", "scope": "test", "companies": ["LLM_DEMO_LOAN", "LLM_DEMO_DEPOSIT", "LLM_DEMO_UNKNOWN"],
             "roles": ["cfo"], "sources": [{"id": "test-case", "title": "Тестовое решение регулятора", "text": BODY,
             "provenance": "synthetic_test", "minimum_facts": 1, "expected_topics": ["policy_rate"],
             "question": "Какие данные договора нужно уточнить?"}]}
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(cases))
    gateway = ScriptedGateway()
    with httpx.Client(transport=httpx.MockTransport(gateway)) as http:
        result = evaluate(settings, path, root / "backend/fixtures/llm_companies.csv", 1,
                          "data/processed/evaluation.json", True,
                          runtime_factory=lambda f, s: runtime(f, s, gateway, client=http)[0])
    assert result["completed_cases"] == 1 and result["failed_cases"] == 0
    report = json.loads(Path(result["report"]).read_text())
    row = report["results"][0]
    assert row["answer"]["as_of"] == row["card"]["as_of"]
    assert row["evidence"] and row["run"]["llm_calls"]
    assert settings.api_key not in json.dumps(report)
    assert "X-Demo-Session" not in json.dumps(report)
    assert not report["semantic_quality_independently_reviewed"]


def test_gateway_receipt_reservation_is_exclusive_across_workers(store):
    from concurrent.futures import ThreadPoolExecutor
    _, factory, _, settings = store
    settings = replace(settings, budget_mode="gateway_receipt", receipt_rub_per_unit=Decimal("1"))
    def attempt(index):
        try:
            reserve(factory, settings, f"exclusive-{index}", f"run-{index}",
                    settings.models[settings.analysis_model], "receipt-call", f"hash-{index}")
            return "reserved"
        except AnalysisError as exc:
            return exc.code
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(attempt, range(2)))
    assert sorted(results) == ["budget_exceeded", "reserved"]
    with factory() as db:
        budget = db.scalar(select(BudgetDay))
        assert budget.reserved_micros == budget.limit_micros


def test_explicit_new_budget_allocation_preserves_unknown_reservation(store):
    _, factory, _, settings = store
    settings = replace(settings, daily_limit_micros=50_000_000, budget_mode="gateway_receipt", receipt_rub_per_unit=Decimal("1"))
    model = settings.models[settings.analysis_model]
    reserve(factory, settings, "unknown-first", "run-first", model, "first", "first-hash")
    with factory.begin() as db:
        budget = db.scalar(select(BudgetDay))
        budget.limit_micros = 100_000_000  # Deliberate operator allocation, never done by a worker.
    reserve(factory, replace(settings, daily_limit_micros=100_000_000), "new-second", "run-second", model, "second", "second-hash")
    with factory() as db:
        assert db.get(LLMCall, "unknown-first").reserved_micros == 50_000_000
        assert db.get(LLMCall, "new-second").reserved_micros == 50_000_000
        assert db.scalar(select(BudgetDay)).reserved_micros == 100_000_000


@pytest.mark.parametrize("content", ['{"ok":true}', '```json\n{"ok":true}\n```', '```\n{"ok":true}\n```'])
def test_json_wrapper_is_supported_without_relaxing_schema(content):
    from app.intelligence.provider import parse_json_object
    assert parse_json_object(content) == {"ok": True}


@pytest.mark.parametrize("content", ['Here is JSON: {"ok":true}', '```json\n{"ok":true}\n```\nextra', '[{"ok":true}]'])
def test_json_embedded_in_prose_or_wrong_root_is_rejected(content):
    from app.intelligence.provider import parse_json_object
    with pytest.raises(AnalysisError, match="llm_invalid_json"):
        parse_json_object(content)


def test_unknown_extraction_cannot_be_retried_by_changing_generation_options(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    first = queued_insight(client, headers["DEMO_A"], event_id).json()
    worker, gateway = runtime(factory, settings, ScriptedGateway(timeout=True))
    process_next(factory, worker)
    assert client.get("/api/v1/jobs/" + first["id"], headers=headers["DEMO_A"]).json()["status"] == "failed"
    changed = replace(settings, models={k: replace(m, reasoning_enabled=False) for k, m in settings.models.items()})
    second = queued_insight(client, headers["DEMO_A"], event_id, key="changed-options-001").json()
    worker, second_gateway = runtime(factory, changed)
    process_next(factory, worker)
    job = client.get("/api/v1/jobs/" + second["id"], headers=headers["DEMO_A"]).json()
    assert job["status"] == "failed" and job["error"] == "extraction_outcome_unknown"
    assert second_gateway.seen == []


@pytest.mark.parametrize("mode", ["json_object", "json_schema"])
def test_native_structured_output_wire_format_and_strict_schema(store, mode):
    from app.intelligence.schemas import ExtractionDraft
    _, factory, _, settings = store
    settings = replace(settings, models={k: replace(m, response_mode=mode) for k, m in settings.models.items()})
    def gateway(request):
        body = json.loads(request.content)
        assert body["response_format"]["type"] == mode
        if mode == "json_schema":
            schema = body["response_format"]["json_schema"]
            assert schema["strict"] is True
            assert set(schema["schema"]["$defs"]["Attributes"]["required"]) == set(schema["schema"]["$defs"]["Attributes"]["properties"])
            assert '"default"' not in json.dumps(schema)
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"facts":[]}'}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        result = Provider(factory, settings, client).generate("native-run", "native-schema", settings.analysis_model,
            "Extract facts", {"text": BODY}, ExtractionDraft.model_json_schema())
    assert ExtractionDraft.model_validate(result).facts == []


def test_one_schema_repair_is_bounded_and_cannot_execute_invalid_tool(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job = queued_insight(client, headers["DEMO_A"], event_id).json()
    scripted = ScriptedGateway()
    def gateway(request):
        response = scripted(request)
        data = response.json()
        content = json.loads(data["choices"][0]["message"]["content"])
        if scripted.seen[-1][0] == "Decision" and scripted.decisions == 1:
            content["decision"]["draft"]["inferences"][0]["id"] = "c1"
            data["choices"][0]["message"]["content"] = json.dumps(content)
        return httpx.Response(200, json=data)
    worker, _ = runtime(factory, settings, gateway)
    process_next(factory, worker)
    result = client.get("/api/v1/jobs/" + job["id"], headers=headers["DEMO_A"]).json()
    assert result["status"] == "succeeded", result
    card = client.get("/api/v1/insights/" + result["result_id"], headers=headers["DEMO_A"]).json()
    run = client.get("/api/v1/runs/" + card["run_id"], headers=headers["DEMO_A"]).json()
    assert len(run["llm_calls"]) == 6
    assert sum(s["action"] == "verify_card_schema" for s in run["steps"]) == 1


def test_materialized_facts_use_only_server_text_and_map_inference_ids():
    from app.intelligence.schemas import ReferencedDraft
    from app.intelligence.evidence import materialize_draft
    evidence = {"news": {"kind": "event_fact", "text": "Исходная новость без пересказа."},
                "company": {"kind": "company_feature", "text": "Кредитный продукт: да."},
                "bank": {"kind": "bank_claim", "text": "Дословная позиция из проверенного обзора.", "position_role": "current_in_document"}}
    draft = ReferencedDraft.model_validate({"fact_evidence_ids": ["bank"],
        "inferences": [{"id": "i1", "text": "Условия договора требуют уточнения.", "evidence_ids": ["news", "company"],
                        "conditions": ["Если есть действующие обязательства"]}], "risks": ["i1"],
        "next_steps": [{"text": "Уточнить условия договора", "basis_inference_ids": ["i1"], "conditions": ["При наличии договора"]}]})
    card, issues = materialize_draft(draft, evidence)
    assert issues == []
    assert [c.text for c in card.claims[:3]] == [evidence[i]["text"] for i in ("bank", "news", "company")]
    assert card.risks == ["c4"] and card.next_steps[0].basis_claim_ids == ["c4"]
    assert verify_draft(card, evidence) == []


def test_rate_decision_topic_is_canonicalized_before_separate_review():
    from types import SimpleNamespace
    from app.intelligence.schemas import ExtractionDraft, ExtractionReview
    source = {"id": "source-test", "version": 1, "title": "Решение регулятора", "text": BODY,
              "received_at": "2026-10-01T00:00:00+00:00"}
    class Candidates:
        provider = SimpleNamespace(settings=SimpleNamespace(extraction_model="fixture-model", analysis_model="fixture-verifier"))
        def ask(self, purpose, schema, system, data, **kwargs):
            if schema is ExtractionDraft:
                return ExtractionDraft.model_validate({"facts": [{"quote": BODY.split('. ')[0]+'.',
                    "attributes": {"kind": "rate_decision", "geography": "RU", "topics": ["inflation", "regulation"]}}]})
            assert schema is ExtractionReview
            assert data["candidates"][0]["attributes"]["topics"] == ["policy_rate"]
            return ExtractionReview(accepted_indexes=[0], issues=[])
    facts, _, _ = extract(source, Candidates())
    assert facts[0]["attributes"]["topics"] == ["policy_rate"]


def test_extraction_repairs_schema_once_then_independently_reviews_quotes(store):
    _, factory, _, settings = store
    scripted = ScriptedGateway()
    extraction_count = 0
    def gateway(request):
        nonlocal extraction_count
        response = scripted(request)
        if scripted.seen[-1][0] == "ExtractionDraft":
            extraction_count += 1
            if extraction_count == 1:
                body = response.json()
                data = json.loads(body["choices"][0]["message"]["content"])
                data["facts"][0]["attributes"]["forecast_horizon"] = "не указано"
                body["choices"][0]["message"]["content"] = json.dumps(data, ensure_ascii=False)
                return httpx.Response(200, json=body)
            assert scripted.seen[-1][1]["text"] == BODY
            assert "forecast_horizon" in scripted.seen[-1][1]["validation_issues"][0]
        return response
    calls = Calls(Provider(factory, settings, httpx.Client(transport=httpx.MockTransport(gateway))), "repair-extraction")
    facts, _, _ = extract({"id": "repair-source", "version": 1, "title": "Решение регулятора", "text": BODY,
                          "received_at": "2026-10-01T00:00:00+00:00"}, calls)
    assert len(facts) == 1 and facts[0]["attributes"]["forecast_horizon"] is None
    assert [title for title, _ in scripted.seen] == ["ExtractionDraft", "ExtractionDraft", "ExtractionReview"]
    with factory() as db:
        assert {c.purpose for c in db.scalars(select(LLMCall))} == {"extraction", "extraction_repair", "extraction_review"}
        assert db.scalar(select(BudgetDay)).reserved_micros == 0


def test_noneconomic_candidate_is_dropped_before_paid_review():
    from types import SimpleNamespace
    from app.intelligence.schemas import ExtractionDraft
    source = {"id": "greeting-source", "version": 1, "title": "Поздравление", "text": "Поздравляем команду с праздником!",
              "received_at": "2026-10-01T00:00:00+00:00"}
    class Candidates:
        provider = SimpleNamespace(settings=SimpleNamespace(extraction_model="fixture-model", analysis_model="fixture-verifier"))
        def ask(self, purpose, schema, system, data, **kwargs):
            assert purpose == "extraction" and schema is ExtractionDraft
            return ExtractionDraft.model_validate({"facts": [{"quote": source["text"], "attributes": {"topics": ["culture"]}}]})
    facts, coverage, _ = extract(source, Candidates())
    assert facts == [] and coverage["complete"]


@pytest.mark.parametrize("persistent,max_calls,expected_calls,expected_facts", [
    (False, 8, 4, 1), (True, 8, 4, 0), (False, 3, 2, 0)])
def test_wrong_extraction_attributes_are_repaired_once_and_independently_rechecked(
        store, persistent, max_calls, expected_calls, expected_facts):
    _, factory, _, settings = store
    settings = replace(settings, max_calls=max_calls)
    text = "В России потребительские цены выросли в отчётном месяце."
    seen = []
    extractions = 0
    def gateway(request):
        nonlocal extractions
        body = json.loads(request.content)
        schema = json.loads(body["messages"][0]["content"].split("schema:\n", 1)[1])
        data = json.loads(body["messages"][1]["content"])["untrusted_data"]
        seen.append(schema["title"])
        if schema["title"] == "ExtractionDraft":
            extractions += 1
            if extractions > 1:
                assert "wrong_fact_kind" in data["validation_issues"] and data["text"] == text
            wrong = extractions == 1 or persistent
            result = {"facts": [{"quote": text, "attributes": {
                "topics": ["policy_rate"] if wrong else ["inflation"], "geography": "RU",
                "kind": "rate_decision" if wrong else "report"}}]}
        else:
            wrong = data["candidates"][0]["attributes"]["kind"] != "report"
            result = {"accepted_indexes": [] if wrong else [0], "issues": ["wrong_fact_kind"] if wrong else []}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 100, "completion_tokens": 80}})
    calls = Calls(Provider(factory, settings, httpx.Client(transport=httpx.MockTransport(gateway))), "attribute-repair")
    facts, _, _ = extract({"id": "inflation-source", "version": 1, "title": "Инфляция", "text": text,
                          "received_at": "2026-10-01T00:00:00+00:00"}, calls)
    assert len(seen) == expected_calls and len(facts) == expected_facts
    if facts:
        assert facts[0]["attributes"]["topics"] == ["inflation"] and facts[0]["attributes"]["kind"] == "report"
        assert facts[0]["text"] == text
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(LLMCall)) == expected_calls
        assert db.scalar(select(BudgetDay)).reserved_micros == 0


@pytest.mark.parametrize("quote,text,expected", [
    ("Банк России сохранил ставку.", "Банк России сохранил ставку.", "RU"),
    ("ФРС сохранила ставку.", "ФРС сохранила ставку.", "US"),
    ("Решение по ставке принято.", "Текст по-русски без названия регулятора.", None),
    ("Решение по ставке принято.", "Банк России и ФРС объявили свои решения.", None),
])
def test_rate_jurisdiction_requires_unambiguous_explicit_regulator(quote, text, expected):
    from app.intelligence.extraction import named_rate_jurisdiction
    assert named_rate_jurisdiction(quote, text) == expected


@pytest.mark.parametrize("fact_id", ["missing", "raw-pdf"])
def test_unreviewed_or_unknown_fact_reference_is_not_materialized(fact_id):
    from app.intelligence.schemas import ReferencedDraft
    from app.intelligence.evidence import materialize_draft
    draft = ReferencedDraft.model_validate({"fact_evidence_ids": [fact_id], "inferences": []})
    card, issues = materialize_draft(draft, {"raw-pdf": {"kind": "document_chunk", "text": "OCR without span review"}})
    assert card is None and issues == ["fact_reference_requires_event_company_or_reviewed_bank_claim"]


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
    assert len(gateway.seen) == 5
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
    assert len(gateway.seen) == 5


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
    assert len(gateway.seen) == 7


def test_macro_forecast_cannot_publish_reported_contract_stability_even_if_reviewer_agrees(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings, ScriptedGateway(bad_contract_stability=True))
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "succeeded", job
    card = client.get("/api/v1/insights/" + job["result_id"], headers=headers["DEMO_A"]).json()
    assert gateway.decisions == 2 and gateway.reviews == 1
    inference = next(c for c in card["claims"] if c["kind"] == "model_inference")
    assert "поступление авансов" in inference["text"] and "неизменном" not in inference["text"]
    run = client.get("/api/v1/runs/" + card["run_id"], headers=headers["DEMO_A"]).json()
    checks = [s for s in run["steps"] if s["action"] == "verify_card"]
    assert any("unsupported_company_contract_stability" in issue for issue in checks[0]["issues"])
    assert checks[-1]["issues"] == []


def test_rate_construction_business_focus_reaches_author_and_independent_reviewer(store):
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings)
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job['status'] == 'succeeded', job
    author = next(data for title, data in gateway.seen if title in {'Decision', 'FinalDecision'})
    reviewer = next(data for title, data in gateway.seen if title in {'VerificationDecision', 'FinalVerificationDecision'})
    assert author['business_focus'] == reviewer['business_focus']
    assert author['business_focus']['channel'] == 'customer_investment_demand'
    assert author['business_focus']['company_ids']
    assert author['business_focus']['unknowns']
    assert author['business_focus']['max_inferences'] == 1
    assert author['inference_basis']['company_ids'] == author['business_focus']['company_ids']
    assert not any('Кредитный продукт' in e['text'] for e in author['evidence'])
    assert not any('Кредитный продукт' in e['text'] for e in reviewer['evidence'])
    card = client.get('/api/v1/insights/' + job['result_id'], headers=headers['DEMO_A']).json()
    run = client.get('/api/v1/runs/' + card['run_id'], headers=headers['DEMO_A']).json()
    focus_step = next(s for s in run['steps'] if s['action'] == 'select_business_focus')
    assert focus_step['result']['channel'] == 'customer_investment_demand'
    assert all(i.startswith('feature_') for i in focus_step['result']['company_ids'])
    initial_searches = [s for s in run['steps'] if s.get('origin') == 'initial_context']
    assert all('пересмотра кредитной ставки' not in s['arguments']['query'] for s in initial_searches)
    check = next(s for s in run['steps'] if s['action'] == 'verify_card')
    assert check['candidate_stage'] == 'pre_publication'
    assert check['candidate']['claims'] == card['claims']


@pytest.mark.parametrize("issue", ["missing_business_mechanism", "unsupported_transmission", "off_focus_inference"])
@pytest.mark.parametrize("persistent", [False, True])
def test_business_quality_verdict_requires_repair_or_prevents_publication(store, issue, persistent):
    """Protocol regression: scripted verdicts test enforcement, not model accuracy."""
    client, factory, headers, settings = store
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings, ScriptedGateway(business_issue=issue, persistent_business_issue=persistent))
    process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert gateway.reviews == gateway.decisions == 2
    assert len(gateway.seen) <= settings.max_calls
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Insight)) == (0 if persistent else 1)
    if persistent:
        assert job["status"] == "failed" and job["error"] == "card_validation_failed"
    else:
        assert job["status"] == "succeeded", job
        card = client.get("/api/v1/insights/" + job["result_id"], headers=headers["DEMO_A"]).json()
        run = client.get("/api/v1/runs/" + card["run_id"], headers=headers["DEMO_A"]).json()
        checks = [s for s in run["steps"] if s["action"] == "verify_card"]
        assert issue in checks[0]["issues"] and checks[-1]["issues"] == []
        inference = next(c for c in card["claims"] if c["kind"] == "model_inference")
        assert "поступление авансов" in inference["text"]


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
    assert gateway.seen[-1][0] == "FinalVerificationDecision"


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
    assert "partial_extraction_bounded_source_windows" in card["limitations"]
    assert card["extraction_coverage"]["complete"] is False
    assert sum(w["end"] - w["start"] for w in card["extraction_coverage"]["windows"]) < len(body)


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
    with factory.begin() as db:
        event = db.get(Event, event_id)
        db.get(SourceDocument, event.source_id).published_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
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
    assert item["relevance_grade"] == 4 and item["relevance_selected"] is True
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
    assert response.json()["items"] == []  # Raw mentions cannot enter the grade 4/5 feed.
    assert [item["event_id"] for item in response.json()["extraction_candidates"]] == [event_id]
    assert response.json()["trace"]["unassessed"] == 1
    assert response.json()["trace"]["model_calls_on_read"] == 0


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
def test_real_bank_index_and_local_e5_with_scripted_generation(store, tmp_path):
    pytest.importorskip("numpy")
    pytest.importorskip("onnxruntime")
    client, factory, headers, settings = store
    root = Path(__file__).resolve().parents[2]
    settings = replace(settings, bank_index=str(root / "data/processed/bank-index-v1"),
                       model_dir=str(root / "data/models/e5-base"), model_manifest=str(root / "knowledge/embedding-model.json"))
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, gateway = runtime(factory, settings, fake_e5=False)
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
    assert len(gateway.seen) == 6
    assert engine.encoder is not None
    trace = client.get("/api/v1/runs/" + card["run_id"], headers=headers["DEMO_A"]).json()
    assert {c["agent_role"] for c in trace["llm_calls"]} == {"extractor", "relevance_judge", "bank_researcher", "analyst", "verifier"}
    assert any(s["action"] == "agent_handoff" and s["agent_role"] == "bank_researcher" for s in trace["steps"])
    status = client.get("/api/v1/llm/status", headers=headers["DEMO_A"]).json()
    assert status["agent_network"]["roles"] == ["bank_researcher", "analyst", "verifier"]
    # A newly deployed worker must preserve the parent's reviewed bank evidence
    # even when its current index cannot be loaded. The API status stays honest.
    question = client.post(f"/api/v1/insights/{card['id']}/questions",
        headers={**headers["DEMO_A"], "Idempotency-Key": "snapshot-bank-question"},
        json={"question": "Какие условия договора следует уточнить?"}).json()
    fresh, question_gateway = runtime(factory, replace(settings, bank_index=str(tmp_path / "new-missing-index")))
    process_next(factory, fresh)
    done = client.get("/api/v1/jobs/" + question["id"], headers=headers["DEMO_A"]).json()
    assert done["status"] == "succeeded", done
    answer = client.get("/api/v1/answers/" + done["result_id"], headers=headers["DEMO_A"]).json()
    assert answer["bank_view"]["status"] == "found"
    assert any(c["kind"] == "bank_view" for c in answer["claims"])
    assert answer["profile_version"] == card["profile_version"] and answer["as_of"] == card["as_of"]
    assert not any(title == "ExtractionDraft" for title, _ in question_gateway.seen)


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
