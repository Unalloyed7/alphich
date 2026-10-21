"""A forecast and a credit flag cannot establish the path of a company's loan terms."""
from copy import deepcopy

import pytest

from app.intelligence.evidence import verify_draft
from app.intelligence.financial_grounding import business_focus, focus_issues
from app.intelligence.schemas import CardDraft

REPORTED = ("Если у компании есть действующий кредитный договор, условия финансирования могут "
            "сохраняться в неизменном виде до конца года в базовом сценарии банка; "
            "стоимость и структуру долга следует уточнить по договору.")


@pytest.fixture
def evidence():
    return {
        "news": {"kind": "event_fact", "text": "Банк России сохранил ключевую ставку."},
        "credit": {"kind": "company_feature", "name": "CL_PRODUCT_LOA_MARK", "value": True,
                   "text": "Кредитный продукт: да."},
        "forecast": {"kind": "bank_claim", "text": "Банк прогнозирует ключевую ставку 14% на конец года.",
                     "metric": "policy_rate", "forecast_horizon": "2026-12-31",
                     "position_role": "current_in_document"},
    }


def draft(text):
    return CardDraft.model_validate({"claims": [{"id": "c1", "kind": "model_inference", "text": text,
        "evidence_ids": ["news", "credit", "forecast"],
        "conditions": ["У компании есть подтверждённый признак кредитного продукта",
                       "Фактические условия договора, тип ставки и даты пересмотра не проверены"]}]})


@pytest.mark.parametrize("text", [REPORTED,
    "Кредитные условия компании останутся прежними до конца года по сценарию банка.",
    "Стоимость обслуживания долга может сохраняться на прежнем уровне в базовом сценарии.",
    "Неизвестен размер долга, однако условия кредитования останутся стабильными.",
    "Если у компании есть действующий кредит с плавающей ставкой, сохранение ключевой ставки означает, что стоимость обслуживания не снизится в ближайшей перспективе.",
])
def test_credit_flag_forecast_and_caveats_do_not_support_contract_stability(evidence, text):
    saved = deepcopy(evidence)
    assert verify_draft(draft(text), evidence) == ["c1:unsupported_company_contract_stability"]
    assert evidence == saved


@pytest.mark.parametrize("text", [
    "Если кредит имеет плавающую ставку с привязкой к ключевой, её изменение может влиять на процентную составляющую; условия договора неизвестны.",
    "Условия финансирования компании могут зависеть от сохранения ключевой ставки; тип ставки и сроки пересмотра требуют проверки.",
    "Прогноз банка не означает, что условия финансирования компании останутся неизменными.",
    "Неизвестно, останутся ли условия кредитования компании прежними.",
])
def test_conditional_mechanism_and_explicit_unknowns_are_not_stability_predictions(evidence, text):
    assert verify_draft(draft(text), evidence) == []


def test_exact_attributed_bank_forecast_stays_a_fact(evidence):
    card = CardDraft.model_validate({"claims": [{"id": "c1", "kind": "bank_view",
        "text": evidence['forecast']['text'], "evidence_ids": ["forecast"]}]})
    assert verify_draft(card, evidence) == []


def construction_context(evidence):
    evidence = deepcopy(evidence)
    evidence['news']['attributes'] = {'topics': ['policy_rate']}
    evidence['industry'] = {'kind': 'company_feature', 'name': 'okved_code', 'value': '41.20', 'text': 'ОКВЭД: 41.20.'}
    return evidence


def test_rate_construction_lens_is_an_explicit_customer_hypothesis_not_company_debt(evidence):
    context = construction_context(evidence)
    saved = deepcopy(context)
    focus = business_focus(context)
    assert focus['channel'] == 'customer_investment_demand'
    assert focus['news_ids'] == ['news'] and focus['company_ids'] == ['industry']
    assert 'credit' in focus['not_inference_basis_ids']
    assert focus['hypothesis'].startswith('Если ')
    assert focus['unknowns'] and 'не установлено' in focus['constraints']
    assert context == saved


def test_credit_flag_alone_and_unrelated_sector_or_news_cannot_trigger_construction_lens(evidence):
    assert business_focus(evidence) is None
    context = construction_context(evidence)
    context['industry']['value'] = '62.01'
    assert business_focus(context)['channel'] == 'customer_it_demand'
    context = construction_context(evidence)
    context['news']['attributes']['topics'] = ['sports']
    assert business_focus(context) is None


def test_direct_question_about_credit_is_not_diverted_to_customer_demand(evidence):
    context = construction_context(evidence)
    assert business_focus(context, question='Как решение влияет на плавающую ставку по кредиту?') is None
    assert focus_issues(draft('Если кредит привязан к ключевой ставке, её изменение может влиять на процентную составляющую.'),
                        context, question='Как решение влияет на кредит?') == []


def test_related_rate_event_cannot_change_the_focus_of_an_unrelated_current_story(evidence):
    context = construction_context(evidence)
    context['news']['event_id'] = 'related-event'
    assert business_focus(context, event_id='current-event') is None
    assert business_focus(context, event_id='related-event')['news_ids'] == ['news']


def test_demand_hypothesis_requires_industry_basis_and_does_not_use_credit_flag_as_exposure(evidence):
    context = construction_context(evidence)
    card = draft('Если запуск проектов заказчиков зависит от кредитов, дорогие кредиты могут откладывать запуск и поступление авансов строительной компании.')
    assert set(focus_issues(card, context)) == {'c1:business_focus_needs_industry_basis',
                                              'c1:credit_flag_does_not_establish_company_financing'}
    card.claims[0].evidence_ids = ['news', 'industry', 'forecast']
    assert focus_issues(card, context) == []


def test_focused_card_cannot_add_competing_extra_inferences(evidence):
    context = construction_context(evidence)
    card = draft('Если заказчики зависят от кредитов, дорогие кредиты могут откладывать запуск проектов.')
    card.claims[0].evidence_ids = ['news', 'industry']
    card.claims.append(card.claims[0].model_copy(update={'id': 'c2'}))
    assert focus_issues(card, context) == ['business_focus_requires_one_primary_inference']


@pytest.mark.parametrize('code,channel', [('41.20', 'customer_investment_demand'),
    ('62.01', 'customer_it_demand'), ('46.90', 'customer_demand_and_inventory'),
    ('49.41', 'logistics_demand'), ('25.11', 'investment_demand')])
def test_rate_mechanism_uses_evidenced_business_activity_without_inventing_own_debt(evidence, code, channel):
    context = construction_context(evidence)
    context['industry']['value'] = code
    focus = business_focus(context)
    assert focus['channel'] == channel and focus['hypothesis'].startswith('Если')
    assert focus['company_ids'] == ['industry'] and 'credit' in focus['not_inference_basis_ids']


def test_inflation_headline_cannot_be_replaced_with_historical_rate_background(evidence):
    context = construction_context(evidence)
    context['inflation'] = {'kind': 'event_fact', 'text': 'Годовая инфляция замедлилась до 7,15%.',
                            'attributes': {'topics': ['inflation']}}
    title = 'Инфляция в России замедлилась до 7,15%'
    focus = business_focus(context, event_title=title)
    assert focus['driver'] == 'inflation' and focus['news_ids'] == ['inflation']
    bad = draft('Если заказчики зависят от кредитов, высокая ставка может ограничивать новые заказы.')
    bad.claims[0].evidence_ids = ['news', 'industry', 'forecast']
    assert focus_issues(bad, context, event_title=title) == ['c1:business_focus_needs_primary_news_basis']
    del context['inflation']
    assert business_focus(context, event_title=title)['max_inferences'] == 0
