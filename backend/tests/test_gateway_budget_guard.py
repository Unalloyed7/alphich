"""No paid requests: real shared-ledger transactions and scripted gateway budget."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import json

import httpx
import pytest
from sqlalchemy import select

from app.core import Base, make_session_factory
from app.intelligence.guarded_provider import GuardedGatewayProvider, observe_and_release, verified_budget
from app.intelligence.provider import reserve
from app.intelligence.settings import AnalysisError, Model, Settings
from app.intelligence.storage import BudgetDay, GatewayBudgetObservation, GatewayReserveRelease, LLMCall


@pytest.fixture
def setup(tmp_path):
    engine, factory = make_session_factory('sqlite:///' + str(tmp_path / 'guard.db'))
    Base.metadata.create_all(engine)
    model = Model('permitted-model', 'glm-5.3-flash')
    settings = Settings(base_url='https://gateway.example.invalid/v1', api_key='fixture-secret',
        extraction_model=model.id, analysis_model=model.id, models={model.id: model},
        budget_mode='gateway_receipt', receipt_rub_per_unit=Decimal('1'))
    day = datetime.now(timezone.utc).date().isoformat()
    with factory.begin() as db:
        db.add(BudgetDay(day=day, limit_micros=500_000_000, spent_micros=19_000_000, reserved_micros=481_000_000))
        db.add(LLMCall(id='old-timeout', run_id='old-run', day=day, model=model.id, purpose='step',
            request_hash='original', status='failed', reserved_micros=481_000_000,
            error='llm_timeout_outcome_unknown'))
    yield factory, settings, day
    engine.dispose()


def budget_info(**changes):
    end = (datetime.now(timezone.utc) + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
    return {'info': {'max_budget': 500, 'spend': 19.43, 'budget_duration': '1d',
                     'budget_reset_at': end.isoformat(), **changes}}


def test_guard_resumes_new_paid_calls_without_inventing_old_cost_or_repeating_it(setup):
    factory, settings, day = setup
    methods = []
    def gateway(request):
        methods.append(request.method)
        if request.method == 'GET':
            return httpx.Response(200, json=budget_info())
        with factory() as db:
            current = db.scalar(select(LLMCall).where(LLMCall.id != 'old-timeout'))
            assert current.reserved_micros == 480_570_000
        return httpx.Response(200, headers={'x-litellm-response-cost': '0.1'}, json={
            'choices': [{'message': {'content': '{"ok":true}'}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})
    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        provider = GuardedGatewayProvider(factory, settings, client, release_call_ids=('old-timeout',))
        assert provider.generate('new-run', 'step', 'permitted-model', 'JSON', {}, {}) == {'ok': True}
        with pytest.raises(AnalysisError, match='previous_call_outcome_unknown'):
            reserve(factory, settings, 'old-timeout', 'old-run', settings.models['permitted-model'], 'step', 'original')
    assert methods == ['GET', 'POST']
    with factory() as db:
        old = db.get(LLMCall, 'old-timeout')
        assert old.cost_micros is None and old.input_tokens is None and old.result is None
        assert old.status == 'failed' and old.reserved_micros == 0
        budget = db.get(BudgetDay, day)
        assert budget.spent_micros == 19_530_000 and budget.reserved_micros == 0
        audit = db.get(GatewayReserveRelease, old.id)
        assert audit.released_micros == 481_000_000
        observation = db.get(GatewayBudgetObservation, audit.observation_id)
        assert observation.added_spend_micros == 430_000
        assert settings.api_key not in json.dumps(observation.snapshot)


@pytest.mark.parametrize('changes', [dict(max_budget=501), dict(spend='NaN'), dict(spend=True),
    dict(budget_duration='30d'), dict(budget_reset_at='2026-01-01T00:00:00+00:00'),
    dict(budget_reset_at='2026-10-25T00:00:00'), dict(blocked=True), dict(status='revoked')])
def test_invalid_or_unverified_budget_never_releases_or_dispatches(setup, changes):
    factory, settings, day = setup
    seen = []
    def gateway(request):
        seen.append(request.method)
        return httpx.Response(200, json=budget_info(**changes))
    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        with pytest.raises(AnalysisError, match='gateway_daily_budget_not_verified'):
            GuardedGatewayProvider(factory, settings, client, release_call_ids=('old-timeout',)).generate(
                'new', 'step', 'permitted-model', 'JSON', {}, {})
    assert seen == ['GET']
    with factory() as db:
        assert db.get(BudgetDay, day).reserved_micros == 481_000_000
        assert db.get(GatewayReserveRelease, 'old-timeout') is None


def test_concurrent_release_is_once_and_never_releases_inflight_request(setup):
    factory, settings, day = setup
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=budget_info()))) as client:
        snapshots = [verified_budget(settings, client) for _ in range(2)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda snapshot: observe_and_release(factory, snapshot, ('old-timeout',)), snapshots))
    with factory.begin() as db:
        assert db.get(BudgetDay, day).reserved_micros == 0
        assert db.get(BudgetDay, day).spent_micros == 19_430_000
        db.add(LLMCall(id='inflight', run_id='active-run', day=day, model='permitted-model',
            purpose='step', request_hash='hash', status='reserved', reserved_micros=100))
        db.get(BudgetDay, day).reserved_micros = 100
    with pytest.raises(AnalysisError, match='gateway_guard_release_not_eligible'):
        observe_and_release(factory, {**snapshots[0], 'observed_at': 1}, ('inflight',))


def test_upstream_limit_or_network_failure_stops_before_paid_call(setup):
    factory, settings, _ = setup
    for code, payload, expected in [(401, {}, 'gateway_daily_budget_not_verified'),
                                    (200, budget_info(spend=500), 'budget_exceeded')]:
        with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(code, json=payload))) as client:
            with pytest.raises(AnalysisError, match=expected):
                GuardedGatewayProvider(factory, settings, client, release_call_ids=('old-timeout',)).generate(
                    'new', 'step', 'permitted-model', 'JSON', {}, {})


def test_only_transient_read_only_budget_lookup_retries(setup):
    _, settings, _ = setup
    statuses = iter((503, 503, 200))
    methods = []
    def gateway(request):
        methods.append(request.method)
        return httpx.Response(next(statuses), json=budget_info())
    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        assert verified_budget(settings, client)['spend_micros'] == 19_430_000
    assert methods == ['GET', 'GET', 'GET']


def test_opt_in_teacher_release_preserves_other_purposes_and_never_retries_failed_calls(setup):
    factory, settings, day = setup
    with factory.begin() as db:
        old = db.get(LLMCall, 'old-timeout')
        old.purpose = 'relevance_teacher_v2'
        old.reserved_micros -= 100
        db.add(LLMCall(id='analyst-unknown', run_id='analyst-run', day=day, model='permitted-model',
            purpose='agent_final', request_hash='other', status='failed', reserved_micros=100,
            error='llm_timeout_outcome_unknown'))
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json=budget_info()))) as client:
        snapshot = verified_budget(settings, client)
    observe_and_release(factory, snapshot, (), failed_teacher_models=tuple(settings.models))
    with factory() as db:
        assert db.get(LLMCall, 'old-timeout').cost_micros is None
        assert db.get(LLMCall, 'old-timeout').reserved_micros == 0
        assert db.get(LLMCall, 'analyst-unknown').reserved_micros == 100
        assert db.get(BudgetDay, day).reserved_micros == 100
    with pytest.raises(AnalysisError, match='previous_call_outcome_unknown'):
        reserve(factory, settings, 'old-timeout', 'old-run', settings.models['permitted-model'],
                'relevance_teacher_v2', 'original')
