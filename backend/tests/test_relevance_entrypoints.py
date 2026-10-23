"""Entrypoints use persisted E5/GLM decisions; no network, E5 inference or paid calls."""
from sqlalchemy import func, select

from app.news import Job, SourceDocument
from app.rss import RssArticle
from app.intelligence.automation import AutomationConfig, AutomationRecord, dispatch_once
from app.intelligence.discovery import analyze_tick, collect_tick
from app.intelligence.relevance import VERSION
from app.intelligence.storage import LLMCall
from test_intelligence import store
from test_news_automation import copies


def scope_config():
    return AutomationConfig(enabled=True, company_ids=("DEMO_A", "DEMO_B"))


def forbid_legacy(*args, **kwargs):
    raise AssertionError("A lexical or legacy grade veto must not decide the new pipeline")


def test_dispatcher_queues_personal_judgment_without_a_keyword_gate(store, monkeypatch):
    _, factory, _, settings = store
    copies(store)
    with factory.begin() as db:
        for source in [*db.scalars(select(SourceDocument)), *db.scalars(select(RssArticle))]:
            source.title, source.body = "Особое событие", "Синонимичный текст без слов из отраслевого словаря."
    monkeypatch.setattr('alphich_kb.personalization.rules.match_reasons', forbid_legacy)
    monkeypatch.setattr('alphich_kb.personalization.grading.grade_event', forbid_legacy)
    monkeypatch.setattr('app.intelligence.relevance.cached_assessment_db', lambda *args: None)
    result = dispatch_once(factory, settings, scope_config())
    assert result['extractions_queued'] == 2 and result['cards_queued'] == 0
    with factory() as db:
        jobs = list(db.scalars(select(Job).where(Job.operation != "publication_ingest")))
        assert {job.operation for job in jobs} == {'news_fact_extraction'}
        assert len({job.idempotency_key for job in jobs}) == 2
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0


def test_only_cached_glm_four_or_five_can_queue_a_card(store, monkeypatch):
    _, factory, _, settings = store
    copies(store)
    def cached(db, profile, *args):
        grade = 5 if profile['id'] == 'DEMO_A' else 3
        return {'status': 'assessed', 'grade_method': VERSION, 'relevance_grade': grade,
                'relevance_selected': grade >= 4, 'relevance_assessment': {'method': VERSION}}
    monkeypatch.setattr('app.intelligence.relevance.cached_assessment_db', cached)
    monkeypatch.setattr('alphich_kb.personalization.grading.grade_event', forbid_legacy)
    result = dispatch_once(factory, settings, scope_config())
    assert result['cards_queued'] == 1 and result['below_threshold'] == 1
    with factory() as db:
        job = db.scalar(select(Job).where(Job.operation == "automatic_insight"))
        assert job.company_id == 'DEMO_A' and job.operation == 'automatic_insight'
        assert {record.grade for record in db.scalars(select(AutomationRecord))} == {3, 5}


def test_semantic_skips_are_ungraded_and_failed_judges_do_not_restart(store, monkeypatch):
    _, factory, _, settings = store
    copies(store)
    def cached(db, profile, *args):
        return {'status': 'semantic_skipped' if profile['id'] == 'DEMO_A' else 'assessment_failed',
                'grade_method': VERSION, 'relevance_grade': None, 'relevance_selected': False,
                'error': 'news_relevance_outcome_unknown', 'relevance_assessment': {}}
    monkeypatch.setattr('app.intelligence.relevance.cached_assessment_db', cached)
    for _ in range(2):
        result = dispatch_once(factory, settings, scope_config())
        assert result['extractions_queued'] == result['cards_queued'] == 0
    with factory() as db:
        assert {record.status for record in db.scalars(select(AutomationRecord))} == {'semantic_skipped', 'blocked'}
        assert {record.grade for record in db.scalars(select(AutomationRecord))} == {None}
        assert db.scalar(select(func.count()).select_from(Job).where(Job.operation != "publication_ingest")) == 0


def test_discovery_candidates_do_not_need_legacy_keywords(store, monkeypatch):
    from test_news_discovery import FEED, copies as discovery_copies, start, detail
    _, factory, _, settings = store
    monkeypatch.setattr('app.rss.load_config', lambda: [FEED])
    discovery_copies(store)
    with factory.begin() as db:
        for source in [*db.scalars(select(SourceDocument)), *db.scalars(select(RssArticle))]:
            source.title, source.body = 'Особое событие', 'Текст без отраслевых ключевых слов.'
    monkeypatch.setattr('alphich_kb.personalization.rules.match_reasons', forbid_legacy)
    monkeypatch.setattr('alphich_kb.personalization.grading.grade_event', forbid_legacy)
    monkeypatch.setattr('app.intelligence.relevance.cached_assessment_db', lambda *args: None)
    ident = start(store).json()['id']
    collect_tick(factory, feeds=[])
    analyze_tick(factory, settings)
    result = detail(store, ident).json()
    assert result['stats']['preselected'] == 1
    assert result['items'][0]['state'] == 'waiting_facts'
    assert result['items'][0]['grade'] is None
    with factory() as db:
        assert db.scalar(select(Job.operation).where(Job.operation != 'publication_ingest')) == 'news_fact_extraction'
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0


def test_feed_reads_only_persisted_glm_grades_and_runs_no_models(store, monkeypatch):
    client, factory, headers, _ = store
    copies(store)
    monkeypatch.setattr('app.intelligence.news_index.encoder_for', forbid_legacy)
    monkeypatch.setattr('alphich_kb.personalization.grading.grade_event', forbid_legacy)
    def cached(db, profile, event, scope, settings, **kwargs):
        assert kwargs == {'allow_historical': True}
        return {'event_id': event['id'], 'event_version': event['version'], 'status': 'assessed',
                'grade_method': VERSION, 'relevance_grade': 5, 'relevance_selected': True,
                'relevance_assessment': {'method': VERSION}, 'published_at': event['published_at'],
                'available_at': event['available_at'], 'assessment_as_of': scope.as_of}
    monkeypatch.setattr('app.intelligence.relevance.cached_assessment_db', cached)
    response = client.get('/api/v1/companies/DEMO_A/feed', headers=headers['DEMO_A'])
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload['mode'] == 'cached_e5_glm' and payload['items'][0]['relevance_grade'] == 5
    assert payload['trace']['model_calls_on_read'] == 0
    assert payload['news_index']['inference_on_read'] is False
    with factory() as db:
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 0
