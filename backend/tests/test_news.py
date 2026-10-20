"""Contract and boundary tests for the real background ingestion path."""
from __future__ import annotations

import csv
import time
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func, select

from app.api import create_app
from app.core import Base, Company, import_companies, make_session_factory
from app.news import Event, Job, SourceDocument
from app.worker import claim_job, fail_claim, process_claim, process_next_job


@pytest.fixture
def app_store(tmp_path, monkeypatch):
    engine, factory = make_session_factory(f"sqlite:///{tmp_path / 'news.db'}")
    Base.metadata.create_all(engine)
    filepath = tmp_path / 'companies.csv'
    with filepath.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, delimiter=';', fieldnames=[
            'SYNTHETIC_U', 'MAINOKVED_CODE', 'REVENUE_COMPANY',
            'MAINOKVED_NAME', 'CL_PRODUCT_LOA_MARK', 'REGION',
        ])
        writer.writeheader()
        writer.writerow(dict(SYNTHETIC_U='U_A', MAINOKVED_CODE='41.20',
                             REVENUE_COMPANY='19000000000', MAINOKVED_NAME='Строительство'))
        writer.writerow(dict(SYNTHETIC_U='U_B', MAINOKVED_CODE='62.01',
                             REVENUE_COMPANY='20000000000', MAINOKVED_NAME='Разработка ПО'))
    import_companies(filepath, factory)
    monkeypatch.setenv('DEMO_COMPANY_IDS', 'U_A,U_B')
    with TestClient(create_app(factory, engine)) as client:
        tokens = {}
        for cid in ('U_A', 'U_B'):
            res = client.post('/api/v1/demo-session', json={'company_id': cid, 'role': 'cfo'})
            assert res.status_code == 201, res.text
            tokens[cid] = {'X-Demo-Session': res.json()['token']}
        yield client, factory, tokens
    engine.dispose()


def story(**overrides):
    return {
        'title': 'Банк России объявил решение по ставке',
        'text': 'Банк России сообщил о сохранении ключевой ставки. Компаниям следует учитывать условия своих договоров.',
        'source_url': 'https://example.org/economics/press-release',
        'published_at': '2026-09-11T12:30:00+03:00',
        **overrides,
    }


def publish(client, header, *, key='test-news-key-001', **changes):
    return client.post('/api/v1/publications', json=story(**changes),
                       headers={**header, 'Idempotency-Key': key})


def test_publication_queue_worker_and_provenance(app_store):
    client, factory, tokens = app_store
    ua = tokens['U_A']
    assert client.post('/api/v1/publications', json=story()).status_code == 401
    resp = publish(client, ua)
    assert resp.status_code == 202, resp.text
    job = resp.json()
    assert job == dict(id=job['id'], status='queued', stage='waiting', result_type='event',
                       result_id=None, error=None)
    assert client.get(f"/api/v1/jobs/{job['id']}", headers=ua).json()['status'] == 'queued'
    assert process_next_job(factory) is True
    assert process_next_job(factory) is False
    after = client.get(f"/api/v1/jobs/{job['id']}", headers=ua).json()
    assert after['status'] == 'succeeded'
    assert after['stage'] == 'stored_raw_event'
    assert after['result_id'].startswith('event_')
    event = client.get(f"/api/v1/events/{after['result_id']}", headers=ua)
    assert event.status_code == 200
    data = event.json()
    assert data['company_id'] == 'U_A'
    assert data['published_at'] == '2026-09-11T09:30:00+00:00'
    assert data['event_at'] is None
    assert data['topics'] == []
    assert data['facts'] == []
    assert data['extraction_status'] == 'pending_extraction'
    assert data['sources'][0]['url_verified'] is False
    assert data['sources'][0]['fetched_at'] is None
    assert data['sources'][0]['received_at']
    source = client.get(f"/api/v1/sources/{data['source_ids'][0]}", headers=ua)
    assert source.status_code == 200
    assert source.json()['text'] == story()['text']
    assert source.json()['published_at'] == '2026-09-11T09:30:00+00:00'
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Job)) == 1
        assert db.scalar(select(func.count()).select_from(Event)) == 1
        assert db.scalar(select(func.count()).select_from(SourceDocument)) == 1


def test_idempotency_and_exact_duplicate_dedup(app_store):
    client, factory, tokens = app_store
    ua = tokens['U_A']
    first = publish(client, ua)
    same = publish(client, ua)
    assert first.status_code == same.status_code == 202
    assert first.json()['id'] == same.json()['id']
    different = publish(client, ua, title='Другая версия заголовка публикации')
    assert different.status_code == 409
    other = publish(client, ua, key='test-news-key-002')
    assert other.status_code == 202
    assert other.json()['id'] != first.json()['id']
    assert process_next_job(factory)
    assert process_next_job(factory)
    one = client.get('/api/v1/jobs/' + first.json()['id'], headers=ua).json()
    two = client.get('/api/v1/jobs/' + other.json()['id'], headers=ua).json()
    assert one['result_id'] == two['result_id']
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(SourceDocument)) == 1
        assert db.scalar(select(func.count()).select_from(Event)) == 1
        assert db.scalar(select(func.count()).select_from(Job)) == 2


def test_tenant_isolation_for_job_event_and_source(app_store):
    client, factory, tokens = app_store
    own = tokens['U_A']
    other = tokens['U_B']
    res = publish(client, own)
    job_id = res.json()['id']
    assert client.get(f'/api/v1/jobs/{job_id}', headers=other).status_code == 404
    assert client.get(f'/api/v1/jobs/{job_id}').status_code == 401
    process_next_job(factory)
    event_id = client.get(f'/api/v1/jobs/{job_id}', headers=own).json()['result_id']
    source_id = client.get(f'/api/v1/events/{event_id}', headers=own).json()['source_ids'][0]
    assert client.get(f'/api/v1/events/{event_id}', headers=other).status_code == 404
    assert client.get(f'/api/v1/sources/{source_id}', headers=other).status_code == 404
    # Same content from another company: separate source, event and job.
    foreign = publish(client, other)
    assert foreign.status_code == 202
    assert foreign.json()['id'] != job_id
    process_next_job(factory)
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(SourceDocument)) == 2
        assert db.scalar(select(func.count()).select_from(Event)) == 2


def test_input_validation_has_no_side_effects(app_store):
    client, factory, tokens = app_store
    headers = tokens['U_A']
    assert client.post('/api/v1/publications', json=story(), headers=headers).status_code == 400
    assert publish(client, headers, text='short').status_code == 422
    assert publish(client, headers, published_at='2026-09-11T10:00:00').status_code == 422
    assert publish(client, headers, source_url='file:///etc/passwd').status_code == 422
    assert publish(client, headers, source_url='http://localhost:9000/news').status_code == 422
    assert publish(client, headers, source_url='http://127.0.0.1/news').status_code == 422
    assert publish(client, headers, source_url='http://169.254.169.254/latest/meta-data').status_code == 422
    assert publish(client, headers, source_url='https://user:pass@site.com/news').status_code == 422
    assert publish(client, headers, title='       ').status_code == 422
    assert publish(client, headers, imaginary_field='a').status_code == 422
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(SourceDocument)) == 0
        assert db.scalar(select(func.count()).select_from(Job)) == 0


def test_worker_reclaims_expired_lease_and_rejects_stale_attempt(app_store):
    client, factory, tokens = app_store
    job_id = publish(client, tokens['U_A']).json()['id']
    first = claim_job(factory)
    assert first == (job_id, 1)
    with factory.begin() as db:
        job = db.get(Job, job_id)
        job.lease_expires_at = int(time.time()) - 1
    newer = claim_job(factory)
    assert newer == (job_id, 2)
    assert process_claim(factory, *first) is False
    assert process_claim(factory, *newer) is True
    with factory() as db:
        assert db.get(Job, job_id).status == 'succeeded'
        assert db.scalar(select(func.count()).select_from(Event)) == 1


def test_failed_job_retries_then_stops(app_store):
    client, factory, tokens = app_store
    job_id = publish(client, tokens['U_A']).json()['id']
    with factory.begin() as db:
        job = db.get(Job, job_id)
        job.source_id = 'src_missing'
    for attempt in range(1, 4):
        assert process_next_job(factory)
        with factory() as db:
            job = db.get(Job, job_id)
            assert job.error == 'ingestion_failed'
            assert job.status == ('failed' if attempt == 3 else 'queued')
        if attempt < 3:
            with factory.begin() as db:
                db.get(Job, job_id).available_at = int(time.time()) - 1
    assert not process_next_job(factory)
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(Event)) == 0


def test_crashed_final_attempt_eventually_fails(app_store):
    client, factory, tokens = app_store
    job_id = publish(client, tokens['U_A']).json()['id']
    with factory.begin() as db:
        job = db.get(Job, job_id)
        job.status = 'running'
        job.attempt_count = 3
        job.lease_expires_at = int(time.time()) - 1
    assert claim_job(factory) is None
    with factory() as db:
        job = db.get(Job, job_id)
        assert job.status == 'failed'
        assert job.error == 'worker_lease_expired'


def test_session_isolation_within_same_company_and_source_alias(app_store):
    client, factory, tokens = app_store
    own = tokens['U_A']
    second_token = client.post('/api/v1/demo-session', json={
        'company_id': 'U_A', 'role': 'ceo'}).json()['token']
    second = {'X-Demo-Session': second_token}
    first = publish(client, own, key='same-company-role-test')
    assert first.status_code == 202
    jid = first.json()['id']
    assert client.get('/api/v1/jobs/' + jid, headers=second).status_code == 404
    assert publish(client, second, key='same-company-role-test').status_code == 202
    while process_next_job(factory):
        pass
    event_id = client.get('/api/v1/jobs/' + jid, headers=own).json()['result_id']
    assert client.get('/api/v1/events/' + event_id, headers=second).status_code == 200


def test_sources_with_different_urls_keep_distinct_provenance(app_store):
    client, factory, tokens = app_store
    ua = tokens['U_A']
    assert publish(client, ua, key='first-origin-news').status_code == 202
    assert publish(client, ua, key='second-origin-news',
                   source_url='https://another.example.com/news').status_code == 202
    assert publish(client, ua, key='third-origin-news',
                   published_at=None).status_code == 202
    while process_next_job(factory):
        pass
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(SourceDocument)) == 3
        assert db.scalar(select(func.count()).select_from(Event)) == 3
        assert db.scalar(select(func.count()).select_from(SourceDocument).where(
            SourceDocument.published_at.is_(None))) == 1


def test_list_events_is_scoped_and_persisted(app_store):
    client, factory, tokens = app_store
    a, b = tokens['U_A'], tokens['U_B']
    assert client.get('/api/v1/companies/U_A/events', headers=a).json() == {'items': []}
    assert client.get('/api/v1/companies/U_B/events', headers=a).status_code == 403
    assert client.get('/api/v1/companies/U_A/events').status_code == 401
    assert client.get('/api/v1/companies/U_A/events?limit=0', headers=a).status_code == 422
    assert client.get('/api/v1/companies/U_A/events?limit=51', headers=a).status_code == 422
    first = publish(client, a, key='first-list-item')
    assert first.status_code == 202
    assert client.get('/api/v1/companies/U_A/events', headers=a).json() == {'items': []}
    assert process_next_job(factory)
    events = client.get('/api/v1/companies/U_A/events', headers=a).json()['items']
    assert len(events) == 1
    assert events[0]['title'] == story()['title']
    assert events[0]['extraction_status'] == 'pending_extraction'
    assert client.get('/api/v1/companies/U_B/events', headers=b).json()['items'] == []
    assert publish(client, a, key='second-list-item', title='Вторая публикация')
    assert process_next_job(factory)
    assert len(client.get('/api/v1/companies/U_A/events?limit=1', headers=a).json()['items']) == 1
    assert len(client.get('/api/v1/companies/U_A/events?limit=50', headers=a).json()['items']) == 2


def test_list_jobs_recovers_session_history_without_cross_session_access(app_store):
    client, factory, tokens = app_store
    a, b = tokens['U_A'], tokens['U_B']
    url = '/api/v1/jobs'
    assert client.get(url).status_code == 401
    assert client.get(url, headers=a).json() == {'items': []}
    assert client.get(url + '?limit=0', headers=a).status_code == 422
    assert client.get(url + '?limit=51', headers=a).status_code == 422

    first = publish(client, a, key='job-list-first')
    assert first.status_code == 202
    first_id = first.json()['id']
    assert process_next_job(factory)
    items = client.get(url, headers=a).json()['items']
    assert len(items) == 1
    assert items[0]['id'] == first_id
    assert items[0]['status'] == 'succeeded'
    assert items[0]['source_title'] == story()['title']
    assert isinstance(items[0]['created_at'], int)

    other_session = client.post('/api/v1/demo-session', json={'company_id': 'U_A', 'role': 'ceo'})
    assert other_session.status_code == 201
    other_headers = {'X-Demo-Session': other_session.json()['token']}
    assert client.get(url, headers=other_headers).json() == {'items': []}
    assert client.get(url, headers=b).json() == {'items': []}

    second = publish(client, a, key='job-list-second', title='Второе событие о ставке')
    assert second.status_code == 202
    assert len(client.get(url + '?limit=1', headers=a).json()['items']) == 1
    ids = {item['id'] for item in client.get(url, headers=a).json()['items']}
    assert ids == {first_id, second.json()['id']}


def test_event_listing_supports_bounded_pagination(app_store):
    client, factory, tokens = app_store
    a, b = tokens['U_A'], tokens['U_B']
    base = '/api/v1/companies/U_A/events'
    assert client.get(base + '?offset=-1', headers=a).status_code == 422
    assert client.get(base + '?offset=100001', headers=a).status_code == 422
    assert client.get('/api/v1/companies/U_B/events?offset=2', headers=a).status_code == 403
    for idx in range(5):
        resp = publish(client, a, key=f'paginated-news-{idx:03}', title=f'Новость номер {idx + 1}')
        assert resp.status_code == 202
        assert process_next_job(factory)
    full = client.get(base + '?limit=50', headers=a).json()['items']
    first = client.get(base + '?limit=2&offset=0', headers=a).json()['items']
    middle = client.get(base + '?limit=2&offset=2', headers=a).json()['items']
    last = client.get(base + '?limit=2&offset=4', headers=a).json()['items']
    empty = client.get(base + '?limit=2&offset=6', headers=a).json()['items']
    assert [item['id'] for item in first + middle + last] == [item['id'] for item in full]
    assert empty == []
    assert client.get('/api/v1/companies/U_B/events', headers=b).json() == {'items': []}
