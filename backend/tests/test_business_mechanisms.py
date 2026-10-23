"""Business mechanisms need a primary event, activity and explicit unknown exposure."""
from copy import deepcopy

import pytest

from app.intelligence.agent import Tools
from app.intelligence.financial_grounding import business_focus, focus_issues
from app.intelligence.schemas import CardDraft


def context(title, *, topic='sector', code='46.90', extra=()):
    result = {
        'news': {'kind': 'event_fact', 'event_id': 'current', 'text': title,
                 'attributes': {'topics': [topic]}},
        'industry': {'kind': 'company_feature', 'name': 'okved_code', 'value': code},
    }
    result.update({key: {'kind': 'company_feature', 'name': name, 'value': value}
                   for key, name, value in extra})
    return result


@pytest.mark.parametrize('title,topic,channel,expected_data', [
    ('Покупатели увеличили отсрочки оплаты поставщикам', 'sector', 'cash_conversion', 'дебиторки'),
    ('Изменились сроки доставки грузов', 'logistics', 'delivery_continuity', 'маршруты'),
    ('Введён инвестиционный вычет для модернизации оборудования', 'regulation', 'investment_feasibility', 'инвестиционный план'),
    ('Сталь подорожала на внутреннем рынке', 'commodities', 'input_cost_margin', 'Ресурсная спецификация'),
    ('На рынке усилился дефицит кадров', 'labor', 'labor_costs', 'вакансии'),
    ('Изменились обязательные требования к маркировке', 'regulation', 'operating_requirements', 'статус нормы'),
])
def test_relevant_mechanism_has_provenance_specific_data_and_a_decision_branch(title, topic, channel, expected_data):
    evidence = context(title, topic=topic)
    before = deepcopy(evidence)
    focus = business_focus(evidence, event_id='current', event_title=title)
    assert evidence == before
    assert focus['channel'] == channel
    assert focus['news_ids'] == ['news'] and focus['industry_basis_ids'] == ['industry']
    assert focus['max_inferences'] == 1 and focus['hypothesis'].startswith('Если ')
    assert expected_data in focus['decision_checks'][0]['data']
    assert 'Если' in focus['decision_checks'][0]['decision_branch'] or 'При' in focus['decision_checks'][0]['decision_branch']
    assert focus['unknowns']


def test_employee_band_and_bank_payroll_do_not_establish_actual_wage_costs_or_replace_activity():
    title = 'На рынке усилился дефицит кадров'
    evidence = context(title, topic='labor', code='62.01', extra=[
        ('staff', 'employees_fns_band', '101–250'),
        ('age', 'company_age_band', 'старше пяти лет'),
        ('stage', 'business_stage_label', 'A'),
        ('negative', 'negative_mark_raw', True),
        ('payroll', 'CL_PRODUCT_SAL_MARK', True),
    ])
    focus = business_focus(evidence, event_title=title)
    assert focus['company_ids'] == ['industry', 'staff']
    assert focus['company_context'][0]['company_ids'] == ['staff']
    assert 'Не текущий штат' in focus['company_context'][0]['meaning']
    card = CardDraft.model_validate({'claims': [{'id': 'c1', 'kind': 'model_inference',
        'text': 'Если трудозатраты растут при фиксированной цене, может снижаться маржа.',
        'evidence_ids': ['news', 'staff'], 'conditions': ['Структура трудозатрат неизвестна']}]})
    assert focus_issues(card, evidence, event_title=title) == ['c1:business_focus_needs_industry_basis']
    card.claims[0].evidence_ids.append('industry')
    assert focus_issues(card, evidence, event_title=title) == []


def test_opaque_stage_cannot_trigger_a_focus_without_an_event_or_activity():
    evidence = context('Компания провела встречу', extra=[
        ('stage', 'business_stage_label', 'растущая'), ('age', 'company_age_band', 'до года')])
    assert business_focus(evidence, event_title='Компания провела встречу') is None
    del evidence['industry']
    evidence['news']['text'] = 'Введён инвестиционный вычет для модернизации оборудования'
    assert business_focus(evidence, event_title=evidence['news']['text']) is None


@pytest.mark.parametrize('field', ['company_age_band', 'business_stage_label', 'negative_mark_raw', 'client_type'])
def test_descriptive_or_opaque_mark_cannot_be_the_only_company_basis_for_a_business_effect(field):
    evidence = context('Состоялась встреча предпринимателей', extra=[('mark', field, 'A')])
    card = CardDraft.model_validate({'claims': [{'id': 'c1', 'kind': 'model_inference',
        'text': 'Если компания растёт, встреча может дать ей новые заказы.',
        'evidence_ids': ['news', 'mark'], 'conditions': ['Стадия требует подтверждения']}]})
    assert focus_issues(card, evidence) == ['c1:business_inference_needs_operating_basis']
    assert focus_issues(card, evidence, question='Что известно о стадии компании?') == []


def test_capex_context_keeps_age_stage_and_registration_as_eligibility_questions():
    title = 'Введён инвестиционный вычет для модернизации оборудования'
    evidence = context(title, extra=[('city', 'city_registration', 'Москва'),
        ('age', 'company_age_band', 'до года'), ('stage', 'business_stage_label', 'A')])
    focus = business_focus(evidence, event_title=title)
    assert set(focus['company_ids']) == {'industry', 'city', 'age', 'stage'}
    assert all('Только место регистрации' in c['meaning'] or 'не устанавливают стадию роста' in c['meaning']
               for c in focus['company_context'])
    assert focus['industry_basis_ids'] == ['industry']


def test_unrelated_or_historical_news_cannot_prove_a_logistics_headline():
    title = 'Изменились сроки доставки грузов'
    evidence = context('Ранее сообщалось об изменении сроков доставки грузов', topic='logistics')
    evidence['related'] = {**evidence['news'], 'text': title, 'event_id': 'other'}
    focus = business_focus(evidence, event_id='current', event_title=title)
    assert focus['news_ids'] == [] and focus['max_inferences'] == 0


def test_factor_checks_are_relevant_bounded_and_cannot_reintroduce_excluded_sources():
    tools = object.__new__(Tools)
    tools.registry = {key: {'kind': 'company_feature'} for key in ('staff', 'stage', 'loan')}
    tools.aliases, tools.real_ids = {}, {}
    tools.business_context = {'observed': [], 'hypotheses': [], 'factor_checks': [
        {'factor': 'stage', 'topics': ['capex'], 'company_feature_ids': ['stage']},
        {'factor': 'loan', 'topics': ['labor'], 'company_feature_ids': ['loan']},
    ] + [{'factor': f'staff_{i}', 'topics': ['labor'], 'company_feature_ids': ['staff']} for i in range(6)]}
    result = tools.prompt_business_context({'driver': 'labor', 'not_inference_basis_ids': ['loan']})
    assert [c['factor'] for c in result['factor_checks']] == ['staff_0', 'staff_1', 'staff_2', 'staff_3']
    assert all(c['company_feature_ids'] == [tools.aliases['staff']] for c in result['factor_checks'])
    assert tools.business_context['factor_checks'][0]['factor'] == 'stage'
