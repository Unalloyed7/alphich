"""Durable click -> collection -> shared facts -> grades -> cards; no real model/network calls."""
import time
from types import SimpleNamespace

import pytest
from sqlalchemy import func, select

from app.news import Job
from app.rss import FeedConfig, RssFeedArticle, hashed as feed_hash
from app.intelligence.discovery import DiscoveryRun, analyze_tick, collect_tick
from app.intelligence.storage import AnalysisTask, BudgetDay, Insight, LLMCall
from app.intelligence.worker import process_next
from test_intelligence import store, runtime, ScriptedGateway
from test_news_automation import copies as public_copies


FEED = FeedConfig(id='fixture', url='https://example.org/feed/', article_hosts=['example.org'])


@pytest.fixture(autouse=True)
def current_feed_config(monkeypatch):
    monkeypatch.setattr('app.rss.load_config', lambda: [FEED])


def copies(store):
    ids = public_copies(store)
    with store[1].begin() as db:
        db.add(RssFeedArticle(article_id='a' * 64, feed_id=FEED.id,
            feed_url_hash=feed_hash(FEED.url), seen_at=int(time.time())))
    return ids


def start(store, *, company='DEMO_A', key='discovery-unit-001', target=1, min_grade=4):
    client, _, headers, _ = store
    return client.post('/api/v1/companies/' + company + '/discovery',
        headers={**headers[company], 'Idempotency-Key': key},
        json={'target': target, 'min_grade': min_grade, 'max_candidates': 10})


def detail(store, ident, company='DEMO_A'):
    client, _, headers, _ = store
    return client.get('/api/v1/discovery/' + ident, headers=headers[company])


def drain(store, engine):
    _, factory, _, settings = store
    for _ in range(12):
        analyze_tick(factory, settings)
        process_next(factory, engine)
        with factory() as db:
            if not db.scalar(select(DiscoveryRun.id).where(DiscoveryRun.active_key.is_not(None))):
                return
    with factory() as db:
        raise AssertionError({'runs': [(r.status, r.stage, r.items) for r in db.scalars(select(DiscoveryRun))],
                              'jobs': [(j.operation, j.status, j.error) for j in db.scalars(select(Job))]})


def test_start_is_explicit_scoped_idempotent_and_restores_across_sessions(store):
    client, factory, headers, _ = store
    path = '/api/v1/companies/DEMO_A/discovery'
    assert client.get(path, headers=headers['DEMO_A']).json() == {'run': None}
    assert client.post(path, json={}).status_code == 401
    assert client.post(path, json={}, headers=headers['DEMO_B']).status_code == 403
    assert client.post(path, json={}, headers=headers['DEMO_A']).status_code == 400
    r = start(store)
    assert r.status_code == 202, r.text
    ident = r.json()['id']
    assert start(store).json()['id'] == ident
    assert start(store, target=2).status_code == 409
    # New clicks/session tokens attach to the existing company/role run.
    token = client.post('/api/v1/demo-session', json={'company_id': 'DEMO_A', 'role': 'cfo'}).json()['token']
    assert client.post(path, json={}, headers={'X-Demo-Session': token,
        'Idempotency-Key': 'another-click-001'}).json()['id'] == ident
    assert detail(store, ident, 'DEMO_B').status_code == 404
    ceo = client.post('/api/v1/demo-session', json={'company_id': 'DEMO_A', 'role': 'ceo'}).json()['token']
    assert client.get('/api/v1/discovery/' + ident, headers={'X-Demo-Session': ceo}).status_code == 404
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(DiscoveryRun)) == 1
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0


def test_collector_progress_is_persistent_and_one_failed_feed_does_not_stop_others(store):
    _, factory, _, _ = store
    ident = start(store).json()['id']
    feeds = [SimpleNamespace(id='first'), SimpleNamespace(id='second')]
    seen = []
    def poll(factory, feed, targets, **kwargs):
        assert targets == ['DEMO_A'] and kwargs['force'] is True
        seen.append(feed.id)
        return {'feed_id': feed.id, 'status': 'failed' if feed.id == 'first' else 'collected'}
    assert collect_tick(factory, feeds=feeds, poll=poll)
    assert detail(store, ident).json()['stats']['feeds_checked'] == 1
    assert collect_tick(factory, feeds=feeds, poll=poll)
    value = detail(store, ident).json()
    assert seen == ['first', 'second'] and value['status'] == 'waiting_ingestion'
    assert len(value['sources']) == 2


def test_click_runs_real_queue_and_only_selected_grade_gets_card(store):
    client, factory, headers, settings = store
    copies(store)
    ident = start(store).json()['id']
    collect_tick(factory, feeds=[])
    engine, gateway = runtime(factory, settings)
    drain(store, engine)
    result = detail(store, ident).json()
    assert result['status'] == 'completed' and result['stop_reason'] == 'target_reached', result
    assert result['stats']['completed'] == result['stats']['checked'] == 1
    assert result['items'][0]['grade'] == 4
    assert [name for name, _ in gateway.seen].count('ExtractionDraft') == 1
    assert float(result['cost_rub']) > 0 and result['cost_unknown'] is False
    eid = result['items'][0]['event_id']
    saved = client.get('/api/v1/events/' + eid + '/saved-insight', headers=headers['DEMO_A']).json()
    assert saved['insight']['id'] == result['items'][0]['insight_id']
    assert client.get('/api/v1/events/' + eid + '/saved-insight', headers=headers['DEMO_B']).status_code == 404
    count = len(gateway.seen)
    assert not analyze_tick(factory, settings)
    assert len(gateway.seen) == count


def test_only_five_does_not_promote_a_four_or_pay_for_personal_analysis(store):
    _, factory, _, settings = store
    copies(store)
    ident = start(store, min_grade=5).json()['id']
    collect_tick(factory, feeds=[])
    engine, gateway = runtime(factory, settings)
    drain(store, engine)
    result = detail(store, ident).json()
    assert result['status'] == 'partial' and result['stop_reason'] == 'sources_exhausted'
    assert result['stats']['completed'] == 0 and result['stats']['below_threshold'] == 1
    assert result['items'][0]['grade'] == 4
    assert [name for name, _ in gateway.seen] == ['ExtractionDraft', 'ExtractionReview']
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Insight)) == 0


def test_new_run_reuses_current_saved_card_without_queuing_another_paid_job(store):
    _, factory, _, settings = store
    copies(store)
    first = start(store).json()['id']
    collect_tick(factory, feeds=[])
    engine, gateway = runtime(factory, settings)
    drain(store, engine)
    saved = detail(store, first).json()['items'][0]['insight_id']
    calls = len(gateway.seen)
    second = start(store, key='discovery-reuse-002').json()['id']
    assert second != first
    collect_tick(factory, feeds=[])
    drain(store, engine)
    value = detail(store, second).json()
    assert value['status'] == 'completed' and value['stats']['reused'] == 1
    assert value['items'][0]['insight_id'] == saved
    assert float(value['cost_rub']) == 0 and len(gateway.seen) == calls


@pytest.mark.parametrize('change', [
    {'id': 'replacement'}, {'url': 'https://example.org/new-feed/'},
    {'article_hosts': ['other.example.org']}, {'article_path_prefixes': ['/business/']},
])
def test_removed_feed_or_changed_source_scope_never_starts_paid_work(store, monkeypatch, change):
    _, factory, _, settings = store
    copies(store)
    monkeypatch.setattr('app.rss.load_config', lambda: [FEED.model_copy(update=change)])
    ident = start(store).json()['id']
    collect_tick(factory, feeds=[])
    engine, gateway = runtime(factory, settings)
    drain(store, engine)
    value = detail(store, ident).json()
    assert value['stop_reason'] == 'sources_exhausted' and value['stats']['preselected'] == 0
    assert gateway.seen == []
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Job).where(
            Job.operation.in_(['news_fact_extraction', 'automatic_insight']))) == 0


def test_revoked_source_is_checked_again_before_a_preselected_item_is_queued(store, monkeypatch):
    from datetime import datetime, timezone
    _, factory, _, settings = store
    ids = copies(store)
    ident = start(store).json()['id']
    with factory.begin() as db:
        run = db.get(DiscoveryRun, ident)
        run.status, run.as_of = 'analyzing', datetime.now(timezone.utc).isoformat()
        run.items = [{'event_id': ids[0], 'title': 'Previously selected', 'state': 'pending', 'grade': None}]
    monkeypatch.setattr('app.rss.load_config', lambda: [FEED.model_copy(update={'id': 'replacement'})])
    analyze_tick(factory, settings)
    value = detail(store, ident).json()
    assert value['items'][0]['error'] == 'source_no_longer_configured'
    engine, gateway = runtime(factory, settings)
    assert not process_next(factory, engine) and gateway.seen == []


def test_stop_cancels_unstarted_paid_job_and_does_not_generate_anything(store):
    client, factory, headers, settings = store
    copies(store)
    ident = start(store).json()['id']
    collect_tick(factory, feeds=[])
    analyze_tick(factory, settings)
    assert detail(store, ident).json()['items'][0]['state'] == 'waiting_facts'
    client.post('/api/v1/discovery/' + ident + '/stop', headers=headers['DEMO_A'])
    analyze_tick(factory, settings)
    engine, gateway = runtime(factory, settings)
    assert not process_next(factory, engine)
    assert gateway.seen == [] and detail(store, ident).json()['status'] == 'cancelled'


def test_unknown_model_outcome_stops_and_is_never_automatically_retried(store):
    _, factory, _, settings = store
    copies(store)
    ident = start(store).json()['id']
    collect_tick(factory, feeds=[])
    engine, gateway = runtime(factory, settings, ScriptedGateway(timeout=True))
    drain(store, engine)
    result = detail(store, ident).json()
    assert result['status'] == 'partial' and 'outcome_unknown' in result['stop_reason']
    calls = len(gateway.seen)
    assert calls == 1
    analyze_tick(factory, settings)
    assert not process_next(factory, engine) and len(gateway.seen) == calls


@pytest.mark.parametrize('reason', ['time_limit', 'budget_day_changed', 'budget_exceeded'])
def test_run_stops_at_server_limits_without_queuing_paid_work(store, reason):
    _, factory, _, settings = store
    copies(store)
    ident = start(store).json()['id']
    collect_tick(factory, feeds=[])
    with factory.begin() as db:
        run = db.get(DiscoveryRun, ident)
        if reason == 'time_limit': run.deadline_at = int(time.time()) - 1
        elif reason == 'budget_day_changed': run.budget_day = '2000-01-01'
        else:
            db.add(BudgetDay(day=run.budget_day, limit_micros=settings.daily_limit_micros,
                            spent_micros=settings.daily_limit_micros, reserved_micros=0))
    analyze_tick(factory, settings)
    result = detail(store, ident).json()
    assert result['status'] == 'partial' and result['stop_reason'] == reason
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(AnalysisTask)) == 0


