"""Explicit user reviews persist against a scoped, versioned analysis; no model work."""
from sqlalchemy import func, select

from app.intelligence.storage import AnalysisTask, Insight, InsightFeedback, LLMCall
from test_intelligence import store


def seed(factory):
    with factory.begin() as db:
        db.add(Insight(id='feedback-card', company_id='DEMO_A', role='cfo', event_id='event-test',
            cache_key='feedback-cache', payload={'event_version': 2, 'profile_version': 'profile-original'},
            evidence={}, snapshot={'versions': {'runtime': 'feedback-test'}}, run_id='run-test', created_at=1))


def test_feedback_is_scoped_persistent_and_cannot_override_provenance(store):
    client, factory, headers, _ = store
    seed(factory)
    url = '/api/v1/insights/feedback-card/feedback'
    assert client.get(url).status_code == 401
    assert client.get(url, headers=headers['DEMO_B']).status_code == 404
    assert client.put(url, headers=headers['DEMO_B'], json={'value': 'useful'}).status_code == 404
    assert client.get(url, headers=headers['DEMO_A']).json() == {'feedback': None}
    result = client.put(url, headers=headers['DEMO_A'], json={'value': 'needs_review', 'comment': '  Проверьте применимость к договору.  '})
    assert result.status_code == 200, result.text
    assert result.json()['feedback']['comment'] == 'Проверьте применимость к договору.'
    assert 'session_hash' not in result.text
    assert client.get(url, headers=headers['DEMO_A']).json() == result.json()
    other = {'X-Demo-Session': client.post('/api/v1/demo-session', json={'company_id': 'DEMO_A', 'role': 'cfo'}).json()['token']}
    assert client.get(url, headers=other).json() == {'feedback': None}
    ceo = {'X-Demo-Session': client.post('/api/v1/demo-session', json={'company_id': 'DEMO_A', 'role': 'ceo'}).json()['token']}
    assert client.put(url, headers=ceo, json={'value': 'useful'}).status_code == 404
    for _ in range(2):
        assert client.put(url, headers=headers['DEMO_A'], json={'value': 'useful', 'comment': ''}).status_code == 200
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(InsightFeedback)) == 1
        review = db.scalar(select(InsightFeedback))
        assert review.context['profile_version'] == 'profile-original'
        assert review.context['event_version'] == 2
        assert review.context['versions'] == {'runtime': 'feedback-test'}
        assert review.value == 'useful'
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0
        assert db.scalar(select(func.count()).select_from(AnalysisTask)) == 0
    for body in [{'value': 'invented'}, {'value': 'useful', 'comment': 'x' * 501}, {'value': 'useful', 'context': {'company_id': 'DEMO_B'}}]:
        assert client.put(url, headers=headers['DEMO_A'], json=body).status_code == 422


def test_feedback_respects_current_document_access_policy(store, monkeypatch):
    client, factory, headers, _ = store
    seed(factory)
    with factory.begin() as db:
        db.get(Insight, 'feedback-card').evidence = {'bank': {'document_id': 'restricted-bank-document'}}
    monkeypatch.setenv('BANK_ALLOWED_DOCUMENT_IDS', 'another-document')
    url = '/api/v1/insights/feedback-card/feedback'
    assert client.get(url, headers=headers['DEMO_A']).status_code == 403
    assert client.put(url, headers=headers['DEMO_A'], json={'value': 'useful'}).status_code == 403
