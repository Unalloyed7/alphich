"""Provider regressions use only local SQLite and scripted HTTP transports."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from decimal import Decimal
import json
from threading import Barrier, Event

import httpx
import pytest
from sqlalchemy import event, select

from app.core import Base, make_session_factory
from app.intelligence import provider as provider_module
from app.intelligence.gateway import inspect_gateway
from app.intelligence.provider import Provider, parse_json_object, reserve, settle
from app.intelligence.settings import AnalysisError, Model, Settings
from app.intelligence.storage import BudgetDay, CostReceipt, LLMCall


@pytest.fixture
def ledger(tmp_path):
    engine, factory = make_session_factory(f"sqlite:///{tmp_path / 'provider.db'}")
    Base.metadata.create_all(engine)
    model = Model("unit-model", "qwen3.6-35b-a3b", Decimal("1"), Decimal("2"))
    settings = Settings(base_url="https://gateway.example.invalid/v1", api_key="fixture-private-value",
                        extraction_model=model.id, analysis_model=model.id, models={model.id: model},
                        prices_confirmed=True, daily_limit_micros=1_000_000)
    yield engine, factory, settings, model
    engine.dispose()


def test_concurrent_settlement_charges_one_time(ledger):
    engine, factory, settings, model = ledger
    reserve(factory, settings, "settle-once", "run", model, "step", "hash")
    readers = Barrier(2)
    atomic_claim_seen = Event()

    @event.listens_for(engine, "before_cursor_execute")
    def mark_atomic_claim(connection, cursor, statement, parameters, context, executemany):
        if statement.startswith("UPDATE llm_calls"):
            atomic_claim_seen.set()

    @event.listens_for(engine, "after_cursor_execute")
    def synchronize_unlocked_read(connection, cursor, statement, parameters, context, executemany):
        # Reproduce the old read-then-update race on SQLite, where FOR UPDATE is
        # ignored. An atomic UPDATE claim serializes before reaching this read.
        if statement.startswith("SELECT llm_calls.") and not atomic_claim_seen.is_set():
            readers.wait(timeout=5)

    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(lambda _: settle(factory, "settle-once", model, {"ok": True},
                                           {"prompt_tokens": 10, "completion_tokens": 5}), range(2)))
    finally:
        event.remove(engine, "before_cursor_execute", mark_atomic_claim)
        event.remove(engine, "after_cursor_execute", synchronize_unlocked_read)
    with factory() as db:
        budget = db.scalar(select(BudgetDay))
        assert budget.spent_micros == 20
        assert budget.reserved_micros == 0


@pytest.mark.parametrize("budget_mode", ["token_prices", "gateway_receipt"])
def test_invalid_model_json_is_charged_but_not_cached(ledger, budget_mode):
    _, factory, settings, model = ledger
    settings = replace(settings, budget_mode=budget_mode, receipt_rub_per_unit=Decimal("1"))
    requests = []

    def gateway(request):
        requests.append(request)
        return httpx.Response(200, headers={"x-litellm-response-cost": "0.000020"}, json={
            "choices": [{"message": {"content": "malformed JSON"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        provider = Provider(factory, settings, client)
        with pytest.raises(AnalysisError, match="llm_invalid_json"):
            provider.generate("invalid-json", "step", model.id, "JSON", {}, {})
        with pytest.raises(AnalysisError, match="previous_call_outcome_unknown"):
            provider.generate("invalid-json", "step", model.id, "JSON", {}, {})
    assert len(requests) == 1
    with factory() as db:
        budget = db.scalar(select(BudgetDay))
        call = db.scalar(select(LLMCall))
        assert budget.spent_micros == 20 and budget.reserved_micros == 0
        assert call.status == "failed" and call.result is None
        assert call.input_tokens == 10 and call.output_tokens == 5
        assert (db.get(CostReceipt, call.id) is not None) == (budget_mode == "gateway_receipt")


def test_partial_trickle_response_checks_total_deadline_and_keeps_reserve(ledger, monkeypatch):
    _, factory, settings, model = ledger
    settings = replace(settings, timeout_seconds=5)
    clock = [0]
    blocks = []

    class Trickle(httpx.SyncByteStream):
        def __iter__(self):
            for i in range(10):
                clock[0] += 3
                blocks.append(i)
                yield b" "

    monkeypatch.setattr(provider_module.time, "monotonic", lambda: clock[0])
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=Trickle()))) as client:
        with pytest.raises(AnalysisError, match="llm_timeout_outcome_unknown"):
            Provider(factory, settings, client).generate("trickle", "step", model.id, "JSON", {}, {})
    assert blocks == [0, 1]
    with factory() as db:
        budget = db.scalar(select(BudgetDay))
        call = db.scalar(select(LLMCall))
        assert budget.spent_micros == 0 and budget.reserved_micros == call.reserved_micros
        assert call.status == "failed" and call.cost_micros is None


@pytest.mark.parametrize("content", ['{"value":NaN}', '{"value":Infinity}', '{"id":"first","id":"second"}',
                                    '{"outer":{"id":1,"id":2}}'])
def test_nonstandard_and_ambiguous_json_is_rejected(content):
    with pytest.raises(AnalysisError, match="llm_invalid_json"):
        parse_json_object(content)


def test_gateway_diagnostic_does_not_forward_nested_or_free_text_budget_fields(ledger):
    _, _, settings, model = ledger

    def gateway(request):
        if request.url.path.endswith("/models"):
            return httpx.Response(200, json={"data": [{"id": model.id}]})
        return httpx.Response(200, json={"info": {
            "max_budget": {"key": settings.api_key}, "spend": settings.api_key,
            "budget_duration": settings.api_key, "budget_reset_at": settings.api_key}})

    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        result = inspect_gateway(settings, client)
    assert settings.api_key not in json.dumps(result)
    assert result["upstream_budget"] == dict.fromkeys(("max_budget", "spend", "budget_duration", "budget_reset_at"))


@pytest.mark.parametrize("models_body,key_info_body", [({"data": None}, {}), ({"data": []}, []),
                                                      ({"data": []}, {"info": []})])
def test_gateway_diagnostic_malformed_shapes_have_safe_error(ledger, models_body, key_info_body):
    _, _, settings, _ = ledger

    def gateway(request):
        return httpx.Response(200, json=models_body if request.url.path.endswith("/models") else key_info_body)

    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        with pytest.raises(AnalysisError, match="gateway_diagnostic_invalid_response"):
            inspect_gateway(settings, client)
