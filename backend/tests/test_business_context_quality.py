import copy

import pytest

from alphich_kb.common import KnowledgeError
from alphich_kb.personalization.business import business_context
from app.intelligence.content_quality import content_quality


def profile(code='62.01'):
    return {'id': 'own', 'profile_version': 'v1', 'industry': {'okved_code': code, 'okved_name': 'Основная деятельность'},
            'region_registration': 'Москва', 'products': {'CL_PRODUCT_LOA_MARK': True},
            'features': [{'id': 'industry', 'name': 'okved_code', 'value': code, 'profile_version': 'v1'},
                         {'id': 'region', 'name': 'region_registration', 'value': 'Москва', 'profile_version': 'v1'},
                         {'id': 'loan', 'name': 'CL_PRODUCT_LOA_MARK', 'value': True, 'profile_version': 'v1'},
                         {'id': 'group', 'name': 'group_id', 'value': 'G1', 'profile_version': 'v1'}]}


def test_dossier_keeps_industry_hypotheses_separate_from_observed_exposure():
    p = profile()
    before = copy.deepcopy(p)
    result = business_context(p)
    assert p == before
    assert {h['id'] for h in result['hypotheses']} == {'customer_it_budgets', 'delivery_economics', 'financing_sensitivity'}
    assert all(h['status'] == 'requires_confirmation' and h['possible_problem'].startswith('Если')
               and h['basis_feature_ids'] == (['loan'] if h['id'] == 'financing_sensitivity' else ['industry'])
               for h in result['hypotheses'])
    assert result['market_position']['market_share'] is None
    assert next(o for o in result['observed'] if o['code'] == 'loan_flag')['value'] is True
    assert 'не подтверждает' in next(o for o in result['observed'] if o['code'] == 'registration')['meaning']
    assert 'общий долг' in next(o for o in result['observed'] if o['code'] == 'group_membership')['meaning']
    assert result['context_hash'] == business_context(p)['context_hash']
    assert result['context_hash'] != business_context(profile('41.20'))['context_hash']


def test_dossier_rejects_another_company_feature_and_stale_version():
    for field, value in [('company_id', 'other'), ('profile_version', 'old')]:
        p = profile()
        p['features'][0][field] = value
        with pytest.raises(KnowledgeError):
            business_context(p)


def test_quality_does_not_call_structural_success_semantic_correctness():
    evidence = {'n': {'kind': 'event_fact', 'company_id': 'own', 'text': 'Новость.'},
                'f': {'kind': 'company_feature', 'company_id': 'own', 'text': 'Деятельность.'}}
    card = {'company_id': 'own', 'relevance_grade': 4, 'claims': [
        {'id': 'c1', 'kind': 'news_fact', 'text': 'Новость.', 'evidence_ids': ['n']},
        {'id': 'c2', 'kind': 'model_inference', 'text': 'Если заказчики меняют планы, возможен пересмотр загрузки.',
         'evidence_ids': ['n', 'f'], 'conditions': ['Состав заказчиков неизвестен']}],
        'next_steps': [{'text': 'Сверить портфель заказов.', 'basis_claim_ids': ['c2'], 'conditions': ['Есть портфель']}],
        'verification': {'status': 'passed_with_conditions', 'issues': []}}
    result = content_quality(card, evidence)
    assert result['factual_quote_match']['ratio'] == 1
    assert result['inference_evidence_structure']['ratio'] == 1
    assert result['action_traceability']['ratio'] == 1
    assert result['semantic_correctness'] is None and result['business_usefulness'] is None
    assert result['human_review']['status'] == 'not_recorded'
    bad = copy.deepcopy(card)
    bad['claims'][0]['text'] = 'Выдуманный факт'
    bad['claims'][1]['evidence_ids'] = ['n', 'foreign']
    bad['next_steps'][0]['basis_claim_ids'] = ['c1']
    evidence['foreign'] = {'kind': 'company_feature', 'company_id': 'other', 'text': 'Чужой профиль'}
    result = content_quality(bad, evidence)
    assert all(result[k]['ratio'] == 0 for k in ('factual_quote_match', 'inference_evidence_structure', 'action_traceability'))
    empty = content_quality({'company_id': 'own'}, {})
    assert empty['factual_quote_match']['ratio'] is None
