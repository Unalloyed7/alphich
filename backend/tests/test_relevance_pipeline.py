"""API/worker acceptance for the mandatory E5 -> GLM -> insight decision."""
from dataclasses import replace

import pytest
from sqlalchemy import func, select

from app.intelligence.relevance import NewsRelevanceAssessment
from app.intelligence.settings import AnalysisError, Settings
from app.intelligence.storage import AnalysisRun, BudgetDay, Extraction, Insight, LLMCall
from app.intelligence.worker import process_next
from test_intelligence import ScriptedGateway, queued_insight, runtime, store, stored_event
from test_news_relevance import Encoder


def completed(client, headers, job):
    return client.get('/api/v1/jobs/' + job['id'], headers=headers).json()


def test_semantically_distant_manual_news_never_reaches_paid_extraction_or_insight(store):
    client, factory, headers, settings = store
    event = stored_event(client, factory, headers['DEMO_A'])
    job = queued_insight(client, headers['DEMO_A'], event).json()
    worker, gateway = runtime(factory, settings)
    worker.encoder = Encoder(0.5)
    assert process_next(factory, worker)
    result = completed(client, headers['DEMO_A'], job)
    assert result['status'] == 'succeeded' and result['stage'] == 'news_filtered'
    assert result['result_type'] == 'event' and result['result_id'] == event
    assert gateway.seen == []
    with factory() as db:
        for table in (LLMCall, BudgetDay, Extraction, Insight):
            assert db.scalar(select(func.count()).select_from(table)) == 0
        assessment = db.scalar(select(NewsRelevanceAssessment))
        assert assessment.status == 'skipped' and assessment.grade is None
        assert db.scalar(select(AnalysisRun)).verification['status'] == 'semantic_below_threshold'


@pytest.mark.parametrize('grade', [0, 1, 2, 3, 4, 5])
def test_glm_grade_alone_controls_whether_manual_analysis_generates_an_insight(store, grade):
    client, factory, headers, settings = store
    event = stored_event(client, factory, headers['DEMO_A'])
    job = queued_insight(client, headers['DEMO_A'], event).json()
    worker, gateway = runtime(factory, settings, ScriptedGateway(relevance_grade=grade))
    assert process_next(factory, worker)
    result = completed(client, headers['DEMO_A'], job)
    assert result['status'] == 'succeeded', result
    assert [name for name, _ in gateway.seen][:3] == ['ExtractionDraft', 'ExtractionReview', 'RelevanceJudgment']
    with factory() as db:
        assert db.scalar(select(NewsRelevanceAssessment)).grade == grade
        assert db.scalar(select(func.count()).select_from(Insight)) == (1 if grade >= 4 else 0)
    if grade < 4:
        assert result['stage'] == 'news_filtered' and result['result_type'] == 'event'
        assert len(gateway.seen) == 3  # No bank researcher, analyst or final verifier.
    else:
        assert result['result_type'] == 'insight'
        card = client.get('/api/v1/insights/' + result['result_id'], headers=headers['DEMO_A']).json()
        assert card['relevance_grade'] == grade and card['relevance_selected'] is True
        assert card['semantic_gate']['similarity'] >= settings.news_e5_min_similarity


def test_missing_e5_blocks_paid_work_even_without_a_bank_index(store, monkeypatch):
    client, factory, headers, settings = store
    event = stored_event(client, factory, headers['DEMO_A'])
    job = queued_insight(client, headers['DEMO_A'], event).json()
    worker, gateway = runtime(factory, settings, fake_e5=False)
    monkeypatch.setattr('app.intelligence.news_index.encoder_for', lambda _: None)
    assert process_next(factory, worker)
    result = completed(client, headers['DEMO_A'], job)
    assert result['status'] == 'failed' and result['error'] == 'news_e5_unavailable'
    assert gateway.seen == []
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0


def test_extract_then_insight_reuses_glm_assessment_and_shared_facts(store):
    client, factory, headers, settings = store
    event = stored_event(client, factory, headers['DEMO_A'])
    response = client.post('/api/v1/events/' + event + '/extract-facts', headers={
        **headers['DEMO_A'], 'Idempotency-Key': 'relevance-facts-first'})
    assert response.status_code == 202
    worker, gateway = runtime(factory, settings)
    assert process_next(factory, worker)
    assert completed(client, headers['DEMO_A'], response.json())['stage'] == 'relevance_checked'
    full = queued_insight(client, headers['DEMO_A'], event, key='relevance-card-next').json()
    assert process_next(factory, worker)
    assert completed(client, headers['DEMO_A'], full)['result_type'] == 'insight'
    for schema in ('ExtractionDraft', 'ExtractionReview', 'RelevanceJudgment'):
        assert sum(name == schema for name, _ in gateway.seen) == 1


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -1.01, 1.01, True])
def test_invalid_e5_threshold_cannot_disable_the_gate_silently(value):
    with pytest.raises(ValueError, match='invalid_news_e5_threshold'):
        replace(Settings(), news_e5_min_similarity=value).validate()
