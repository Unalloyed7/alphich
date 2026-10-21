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
from app.intelligence.provider import Provider, parse_json_object, reconcile_zero_http_receipts, reserve, settle
from app.intelligence.settings import AnalysisError, Model, Settings
from app.intelligence.storage import BudgetDay, CostReceipt, CostReconciliation, LLMCall


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


def test_explicit_budget_allocation_preserves_spend_unknown_reserve_and_call_identity(ledger):
    from datetime import datetime, timezone
    _, factory, settings, model = ledger
    settings = replace(settings, daily_limit_micros=100_000_000, budget_mode='gateway_receipt',
                       receipt_rub_per_unit=Decimal('1'))
    day = datetime.now(timezone.utc).date().isoformat()
    with factory.begin() as db:
        db.add(BudgetDay(day=day, limit_micros=10_000_000,
                         spent_micros=9_511_496, reserved_micros=488_504))
        db.add(LLMCall(id='unknown-prior-call', run_id='old-run', day=day, model=model.id,
            purpose='step', request_hash='hash', status='reserved', reserved_micros=488_504))
    for _ in range(2):
        state = provider_module.apply_configured_budget(factory, settings)
        assert state['daily_limit_rub'] == '100.0'
        assert state['spent_rub'] == '9.511496' and state['reserved_rub'] == '0.488504'
    with pytest.raises(AnalysisError, match='previous_call_outcome_unknown'):
        reserve(factory, settings, 'unknown-prior-call', 'old-run', model, 'step', 'hash')
    reserve(factory, settings, 'new-authorized-call', 'new-run', model, 'step', 'new-hash')
    with factory() as db:
        budget = db.get(BudgetDay, day)
        assert budget.limit_micros == 100_000_000 and budget.spent_micros == 9_511_496
        assert db.get(LLMCall, 'unknown-prior-call').reserved_micros == 488_504
        assert db.get(LLMCall, 'unknown-prior-call').status == 'reserved'
        assert db.get(LLMCall, 'new-authorized-call').reserved_micros == 90_000_000
        assert budget.reserved_micros == 90_488_504


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


def test_zero_cost_http_rejection_keeps_failure_and_unknown_tokens_but_releases_money(ledger):
    _, factory, settings, model = ledger
    settings = replace(settings, budget_mode="gateway_receipt", receipt_rub_per_unit=Decimal("1"))
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(
            503, headers={"x-litellm-response-cost": "0"}, json={"error": "unavailable"}))) as client:
        with pytest.raises(AnalysisError, match="llm_http_503"):
            Provider(factory, settings, client).generate("rejection", "step", model.id, "JSON", {}, {})
    with factory() as db:
        call, budget = db.scalar(select(LLMCall)), db.scalar(select(BudgetDay))
        assert call.status == "failed" and call.error == "llm_http_503" and call.result is None
        assert call.input_tokens is None and call.output_tokens is None
        assert call.cost_micros == 0 and budget.spent_micros == 0 and budget.reserved_micros == 0


@pytest.mark.parametrize("receipt", [None, "invalid", "0.1"])
def test_http_rejection_without_proven_zero_retains_reserve(ledger, receipt):
    _, factory, settings, model = ledger
    settings = replace(settings, budget_mode="gateway_receipt", receipt_rub_per_unit=Decimal("1"))
    headers = {} if receipt is None else {"x-litellm-response-cost": receipt}
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(503, headers=headers))) as client:
        with pytest.raises(AnalysisError, match="llm_http_503"):
            Provider(factory, settings, client).generate("rejection", "step", model.id, "JSON", {}, {})
    with factory() as db:
        call, budget = db.scalar(select(LLMCall)), db.scalar(select(BudgetDay))
        assert call.cost_micros is None and budget.reserved_micros == call.reserved_micros


def test_legacy_zero_receipt_reconciliation_is_atomic_and_leaves_other_unknown_calls_held(ledger):
    _, factory, _, _ = ledger
    with factory.begin() as db:
        db.add(BudgetDay(day="2026-10-21", limit_micros=3_000_000, spent_micros=0, reserved_micros=3_000_000))
        for ident, error, amount in [("zero", "llm_http_503", "0"),
                                     ("timeout", "llm_timeout_outcome_unknown", "0"),
                                     ("positive", "llm_http_503", "0.1")]:
            db.add(LLMCall(id=ident, run_id="old-" + ident, day="2026-10-21", model="unit-model", purpose="step",
                request_hash="hash", status="failed", reserved_micros=1_000_000, cost_micros=None,
                input_tokens=None, output_tokens=None, result=None, error=error))
            db.add(CostReceipt(call_id=ident, amount=amount, rub_per_unit="1", source="x-litellm-response-cost"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: reconcile_zero_http_receipts(factory), range(2)))
    assert sum(r["settled_zero_receipts"] for r in results) == 1
    assert reconcile_zero_http_receipts(factory)["settled_zero_receipts"] == 0
    with factory() as db:
        budget = db.get(BudgetDay, "2026-10-21")
        assert budget.limit_micros == 3_000_000 and budget.spent_micros == 0 and budget.reserved_micros == 2_000_000
        assert db.get(LLMCall, "zero").cost_micros == 0
        assert db.get(LLMCall, "zero").status == "failed"
        assert db.get(LLMCall, "timeout").cost_micros is None and db.get(LLMCall, "positive").cost_micros is None
        assert db.get(CostReconciliation, "zero").released_micros == 1_000_000
