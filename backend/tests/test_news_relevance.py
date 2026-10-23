"""Free E5 gate and GLM-owned grade contract; no network or paid calls."""
from dataclasses import replace
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest
from sqlalchemy import func, select

from app.core import Base, make_session_factory
from app.intelligence.relevance import (NewsRelevanceAssessment, RelevanceJudgment, assess,
    cached_assessment, record_gate_skip, semantic_gate)
from app.intelligence.settings import AnalysisError
from alphich_kb.personalization.inputs import SelectionScope


@pytest.fixture
def db(tmp_path):
    engine, factory = make_session_factory(f"sqlite:///{tmp_path / 'relevance.db'}")
    Base.metadata.create_all(engine)
    yield factory
    engine.dispose()


def inputs():
    company = {"id": "C1", "profile_version": "v1", "role": "cfo",
        "industry": {"okved_code": "62.01", "okved_name": "Разработка программного обеспечения"},
        "features": [{"id": "feature_industry", "name": "industry", "profile_version": "v1",
            "value": {"code": "62.01", "name": "Разработка программного обеспечения"}}]}
    body = "Для разработчиков программного обеспечения предложены новые условия работы."
    event = {"id": "EV1", "version": 1, "company_id": "C1", "title": "Новые условия для разработчиков",
        "published_at": "2026-10-23T07:00:00+00:00", "available_at": "2026-10-23T08:00:00+00:00",
        "sources": [{"id": "S1", "company_id": "C1", "version": 1, "text": body,
            "received_at": "2026-10-23T08:00:00+00:00", "published_at": "2026-10-23T07:00:00+00:00"}],
        "facts": [{"id": "fact_1", "text": body, "source_id": "S1", "source_version": "1",
            "source_span": {"start": 0, "end": len(body)}, "source_available_at": "2026-10-23T08:00:00+00:00",
            "verification_status": "verified", "attributes": {"topics": ["regulation"],
                "industry_codes": ["62.01"], "event_status": "proposal"}}]}
    scope = SelectionScope("C1", "2026-10-23T09:00:00+00:00", frozenset({"EV1"}), "cfo")
    settings = SimpleNamespace(analysis_model="organizer-glm", news_e5_min_similarity=0.75)
    return company, event, scope, settings


class Encoder:
    dimension = 2
    space_id = "pinned-e5-test-space"

    def __init__(self, score=0.8):
        self.score, self.queries, self.passages = score, [], []

    def count_tokens(self, text):
        return len(text) // 4 + 2

    def encode_query(self, text):
        self.queries.append(text)
        return np.array([1, 0], dtype=np.float32)

    def encode_passages(self, texts, batch_size=4):
        self.passages.extend(texts)
        return np.asarray([[self.score, sqrt(1 - self.score ** 2)] for _ in texts], dtype=np.float32)


class Judge:
    run_id = "test-run"

    def __init__(self, grade=4, *, fact_ids=None, feature_ids=None, failure=None):
        self.grade, self.seen, self.failure = grade, [], failure
        self.fact_ids = ["fact_1"] if fact_ids is None else fact_ids
        self.feature_ids = ["feature_industry"] if feature_ids is None else feature_ids

    def ask(self, purpose, schema, system, data, **kwargs):
        self.seen.append((purpose, data))
        if self.failure:
            raise AnalysisError(self.failure)
        return schema.model_validate({"grade": self.grade,
            "reason": "Условия применимы к подтверждённой деятельности компании; оценена существенность воздействия.",
            "news_fact_ids": self.fact_ids, "company_feature_ids": self.feature_ids,
            "missing_data": ["Уточнить действующие условия и дату вступления нормы"],
            "required_action": "Проверить применимость требований к деятельности компании"})


@pytest.mark.parametrize("score,expected", [(0.74, "skipped"), (0.75, "passed"), (0.8, "passed")])
def test_gate_uses_cosine_and_inclusive_threshold_not_business_grade(score, expected):
    profile, event, _, _ = inputs()
    encoder = Encoder(score)
    gate = semantic_gate(profile, event, encoder, 0.75)
    assert gate["status"] == expected and gate["similarity"] == pytest.approx(score)
    assert gate["score_is_probability"] is False and gate["coverage_complete"] is True
    assert encoder.passages[0].startswith("passage: ")
    assert all("Россия. Решения Банка России" not in query and "Казначейство" not in query for query in encoder.queries)
    assert gate["matched_facet"]["feature_ids"] == ["feature_industry"]
    assert gate["matched_passage"] == {"source_id": "S1", "source_version": "1", "start": 0, "end": len(event["sources"][0]["text"])}


@pytest.mark.parametrize("bad", [None, "missing_dimensions", "nonfinite", "zero", "wrong_dimension"])
def test_e5_failures_never_fall_back_to_lexical_or_paid_calls(bad):
    profile, event, _, _ = inputs()
    encoder = Encoder()
    if bad is None:
        encoder = None
    elif bad == "missing_dimensions":
        encoder.dimension = None
    else:
        vector = {"nonfinite": [float("nan"), 0], "zero": [0, 0], "wrong_dimension": [1, 0, 0]}[bad]
        encoder.encode_query = lambda _: np.asarray(vector)
    with pytest.raises(AnalysisError, match="news_e5_"):
        semantic_gate(profile, event, encoder)


def test_long_article_gate_records_bounded_evenly_spread_source_coverage():
    profile, event, _, _ = inputs()
    body = "Первый раздел. " + "Разработчики программного обеспечения. " * 1000 + "Последний раздел."
    event["sources"][0]["text"] = body
    event["facts"] = []
    gate = semantic_gate(profile, event, Encoder())
    assert gate["passage_count"] == 16 and gate["total_passages"] > 16
    assert gate["coverage_complete"] is False


def test_skip_is_persisted_without_glm_grade_or_call(db):
    profile, event, scope, settings = inputs()
    calls = Judge()
    gate = semantic_gate(profile, event, Encoder(0.5))
    result = assess(db, profile, event, scope, calls, gate, settings)
    assert result["status"] == "semantic_skipped" and result["relevance_grade"] is None
    assert result["relevance_selected"] is False and calls.seen == []
    assert cached_assessment(db, profile, event, scope, settings) == result
    with db() as session:
        assert session.scalar(select(func.count()).select_from(NewsRelevanceAssessment)) == 1


@pytest.mark.parametrize("grade", range(6))
def test_glm_owns_all_grades_even_when_legacy_rules_would_cap_proposal(db, grade):
    profile, event, scope, settings = inputs()
    calls = Judge(grade)
    gate = semantic_gate(profile, event, Encoder())
    result = assess(db, profile, event, scope, calls, gate, settings)
    assert result["relevance_grade"] == grade
    assert result["relevance_selected"] is (grade >= 4)
    assert calls.seen[0][0] == "news_relevance_judge"
    assert calls.seen[0][1]["as_of"] == scope.as_of
    assert calls.seen[0][1]["news_facts"][0]["text"] == event["sources"][0]["text"]


def test_same_day_repeated_request_reuses_one_judgment_with_immutable_asof(db):
    profile, event, scope, settings = inputs()
    calls = Judge()
    gate = semantic_gate(profile, event, Encoder())
    result = assess(db, profile, event, scope, calls, gate, settings)
    later = replace(scope, as_of="2026-10-23T09:30:00+00:00")
    assert assess(db, profile, event, later, calls, gate, settings) == result
    assert cached_assessment(db, profile, event, later, settings) == result
    assert len(calls.seen) == 1
    with db() as session:
        assert session.get(NewsRelevanceAssessment, result["relevance_assessment_id"]).as_of == scope.as_of
    assert cached_assessment(db, profile, event, replace(scope, as_of="2026-10-24T09:00:00+00:00"), settings) is None
    assert cached_assessment(db, profile, event, scope, SimpleNamespace(analysis_model=settings.analysis_model,
        news_e5_min_similarity=0.8)) is None


@pytest.mark.parametrize("foreign", ["news", "company"])
def test_judge_cannot_reference_foreign_evidence_and_failure_is_not_retried(db, foreign):
    profile, event, scope, settings = inputs()
    calls = Judge(fact_ids=["fact_other"] if foreign == "news" else None,
                  feature_ids=["company_other"] if foreign == "company" else None)
    gate = semantic_gate(profile, event, Encoder())
    with pytest.raises(AnalysisError, match="news_relevance_evidence_outside_scope"):
        assess(db, profile, event, scope, calls, gate, settings)
    blocked = cached_assessment(db, profile, event, scope, settings)
    assert blocked["status"] == "assessment_failed" and blocked["relevance_grade"] is None
    with pytest.raises(AnalysisError, match="news_relevance_outcome_unknown"):
        assess(db, profile, event, replace(scope, as_of="2026-10-23T10:00:00+00:00"), calls, gate, settings)
    assert len(calls.seen) == 1


def test_paid_timeout_is_durable_and_does_not_trigger_fresh_request(db):
    profile, event, scope, settings = inputs()
    calls = Judge(failure="gateway_timeout")
    gate = semantic_gate(profile, event, Encoder())
    with pytest.raises(AnalysisError, match="gateway_timeout"):
        assess(db, profile, event, scope, calls, gate, settings)
    with pytest.raises(AnalysisError, match="news_relevance_outcome_unknown"):
        assess(db, profile, event, scope, calls, gate, settings)
    assert len(calls.seen) == 1


def test_gate_is_bound_to_company_text_and_configuration(db):
    profile, event, scope, settings = inputs()
    gate = semantic_gate(profile, event, Encoder())
    event["title"] = "Другой исходный текст"
    calls = Judge()
    with pytest.raises(AnalysisError, match="news_relevance_gate_mismatch"):
        assess(db, profile, event, scope, calls, gate, settings)
    assert calls.seen == []


def test_high_grade_schema_requires_both_company_and_news_evidence():
    with pytest.raises(ValueError, match="requires news and company evidence"):
        RelevanceJudgment(grade=4, reason="Подтверждение отсутствует", news_fact_ids=[], company_feature_ids=[])


def test_no_verified_facts_can_receive_a_low_glm_grade_without_invented_evidence(db):
    profile, event, scope, settings = inputs()
    event["facts"] = []
    calls = Judge(0, fact_ids=[], feature_ids=[])
    gate = semantic_gate(profile, event, Encoder())
    result = assess(db, profile, event, scope, calls, gate, settings)
    assert result["relevance_grade"] == 0 and not result["relevance_selected"]
    assert calls.seen[0][1]["news_facts"] == []


def test_short_model_references_are_restored_to_immutable_evidence_ids(db):
    profile, event, scope, settings = inputs()
    calls = Judge(fact_ids=["n1"], feature_ids=["f1"])
    result = assess(db, profile, event, scope, calls, semantic_gate(profile, event, Encoder()), settings)
    assert calls.seen[0][1]["news_facts"][0]["id"] == "n1"
    assert calls.seen[0][1]["company_features"][0]["id"] == "f1"
    assert result["matched_fact_ids"] == ["fact_1"] and result["matched_feature_ids"] == ["feature_industry"]


def test_extraction_annotation_version_does_not_invalidate_source_bound_gate(db):
    profile, event, scope, settings = inputs()
    event["facts"] = []
    gate = semantic_gate(profile, event, Encoder())
    _, extracted, _, _ = inputs()
    extracted["version"] = 2
    calls = Judge()
    result = assess(db, profile, extracted, scope, calls, gate, settings)
    assert result["relevance_grade"] == 4 and len(calls.seen) == 1
    assert result["event_version"] == "2"
    assert result["semantic_gate"] == gate
    assert result["assessment_as_of"] == scope.as_of
    assert result["relevance_assessment"]["business_channel"] == result["reason"]
    assert result["relevance_assessment"]["required_action"] is not None


def test_feed_can_show_dated_old_completed_grade_without_reusing_for_generation(db):
    profile, event, scope, settings = inputs()
    calls = Judge()
    result = assess(db, profile, event, scope, calls, semantic_gate(profile, event, Encoder()), settings)
    tomorrow = replace(scope, as_of="2026-10-24T09:00:00+00:00")
    assert cached_assessment(db, profile, event, tomorrow, settings) is None
    historical = cached_assessment(db, profile, event, tomorrow, settings, allow_historical=True)
    assert historical == result and historical["assessment_as_of"] == scope.as_of
    assert len(calls.seen) == 1


def test_historical_feed_mode_does_not_reuse_old_failed_or_semantic_skip(db):
    profile, event, scope, settings = inputs()
    tomorrow = replace(scope, as_of="2026-10-24T09:00:00+00:00")
    calls = Judge(failure="gateway_timeout")
    with pytest.raises(AnalysisError, match="gateway_timeout"):
        assess(db, profile, event, scope, calls, semantic_gate(profile, event, Encoder()), settings)
    assert cached_assessment(db, profile, event, tomorrow, settings, allow_historical=True) is None
    record_gate_skip(db, profile, event, scope, semantic_gate(profile, event, Encoder(0.5)), settings)
    assert cached_assessment(db, profile, event, tomorrow, settings, allow_historical=True) is None


def test_full_company_factors_and_ten_long_facts_fit_actual_32k_wire_budget(db):
    from app.intelligence.provider import input_bytes
    from app.intelligence.relevance import SYSTEM
    from app.intelligence.settings import Model
    profile, event, scope, settings = inputs()
    meaningful = {
        'organization_type': 'Юридическое лицо', 'ownership_form': 'Частная собственность',
        'company_age_band': 'От 5 до 10 лет', 'business_stage_label': 'Зрелый бизнес',
        'negative_mark_raw': 'N', 'employees_fns_band': '101–250', 'employees_spark_band': '101–250',
        'employees_band': '101–250', 'city_registration': 'Москва', 'client_type': 'Корпоративный клиент',
        'revenue_raw': '1000000000', 'revenue_currency': 'RUB', 'revenue_unit': 'RUB',
        'revenue_period': '2025', 'revenue_scope': 'company_or_group_unknown', 'group_id': 'opaque_group',
    }
    for i, (name, value) in enumerate(meaningful.items()):
        profile['features'].append({'id': f'factor_{i}', 'name': name, 'value': value,
            'profile_version': 'v1', 'source_refs': ['fns-source-row-' + 'x' * 1000] * 10,
            'limitations': ['source_date_unknown'] * 10})
    profile.update(revenue_currency='RUB', revenue_period='2025', revenue_scope='company_or_group_unknown',
                   employee_band_status='consistent')
    product_names = ('CL_PRODUCT_LOA_MARK', 'CL_PRODUCT_DEP_MARK', 'CL_PRODUCT_LEA_MARK',
                     'CL_PRODUCT_GAR_MARK', 'FACTORING_MARK', 'CL_PRODUCT_RKO_MARK', 'CL_PRODUCT_ZPP_MARK')
    profile['products'] = {name: True for name in product_names}
    for i, name in enumerate(product_names):
        profile['features'].append({'id': f'product_{i}', 'name': name, 'value': True, 'profile_version': 'v1'})
    payment = {'amount_raw': '1000000', 'currency': 'RUB', 'period': {'year': None, 'end': None},
        'payment_count': 200, 'counterparty_count': 100, 'known_profile_amount_share': 0.9,
        'top_counterparty_share': 0.01, 'shares_denominator_raw': '1000000',
        'limitations': ['payment_purpose_unknown', 'payment_period_end_and_year_unknown'],
        'known_sectors': [{'code': '62.01', 'name': 'Разработка программного обеспечения',
                           'share_of_all_flow': 0.01, 'amount_raw': '10000'}] * 100,
        'top_counterparties': [{'company_id': 'opaque_id' * 20, 'profile_known': True,
            'industry': {'code': '62.01', 'name': 'Разработка программного обеспечения'},
            'share_of_all_flow': 0.01, 'amount_raw': '10000'}] * 100}
    for direction in ('incoming', 'outgoing'):
        profile['features'].append({'id': 'payment_' + direction, 'name': 'payment_' + direction,
            'value': payment, 'profile_version': 'v1', 'method': 'nonself_all_flow_denominator_normalized_duplicates_once'})
    parts = [(f'Событие {i}: ' + 'Условия работы разработчиков программного обеспечения изменились. ' * 35)[:1590]
             for i in range(10)]
    event['sources'][0]['text'] = '\n'.join(parts)
    event['facts'] = []
    offset = 0
    for i, text in enumerate(parts):
        event['facts'].append({'id': f'long_fact_{i}', 'text': text, 'source_id': 'S1', 'source_version': '1',
            'source_span': {'start': offset, 'end': offset + len(text)},
            'source_available_at': '2026-10-23T08:00:00+00:00', 'verification_status': 'verified',
            'attributes': {'topics': ['regulation'], 'industry_codes': ['62.01']}})
        offset += len(text) + 1
    settings.models = {settings.analysis_model: Model(settings.analysis_model, 'glm-5.3-flash', input_limit=32768)}
    calls = Judge(fact_ids=['n1'], feature_ids=['f1'])
    result = assess(db, profile, event, scope, calls, semantic_gate(profile, event, Encoder()), settings)
    data = calls.seen[0][1]
    size = input_bytes(settings.models[settings.analysis_model], SYSTEM, data, RelevanceJudgment.model_json_schema())
    assert size <= 32768
    sent_names = {feature['name'] for feature in data['company_features']}
    assert set(meaningful) | set(product_names) | {'industry', 'payment_incoming', 'payment_outgoing'} <= sent_names
    assert 'features' not in data['company_profile']
    assert all(set(feature) <= {'id', 'name', 'value', 'unit'} for feature in data['company_features'])
    assert data['news_facts'][0]['text'] == parts[0]
    assert data['coverage']['facts_total'] == 10
    assert 1 <= data['coverage']['facts_shown'] < 10
    audit = result['relevance_assessment']['prompt_coverage']
    assert audit['request_bytes'] == size and audit['all_company_factors_retained'] is True
    assert 'partial_judgment_fact_coverage' in result['limitations']
