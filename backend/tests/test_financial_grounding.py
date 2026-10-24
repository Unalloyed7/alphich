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


def test_rate_context_preserves_product_evidence_without_selecting_customer_demand(evidence):
    context = construction_context(evidence)
    saved = deepcopy(context)
    focus = business_focus(context)
    assert focus['channel'] == 'open_business_impact' and focus['driver'] == 'open'
    assert focus['news_ids'] == ['news']
    assert focus['company_ids'] == ['credit', 'industry']
    assert focus['industry_basis_ids'] == ['industry']
    assert focus['not_inference_basis_ids'] == []
    assert focus['selection_policy']['industry_evidence_required'] is False
    assert context == saved


def test_industry_and_known_financial_topic_are_not_required_for_impact_analysis(evidence):
    focus = business_focus(evidence)
    assert focus['company_ids'] == ['credit']
    assert focus['industry_basis_ids'] == []
    context = construction_context(evidence)
    context['news']['attributes']['topics'] = ['unlisted_business_subject']
    assert business_focus(context)['news_ids'] == ['news']


def test_direct_question_about_credit_is_not_diverted_to_customer_demand(evidence):
    context = construction_context(evidence)
    assert business_focus(context, question='Как решение влияет на плавающую ставку по кредиту?') is None
    assert focus_issues(draft('Если кредит привязан к ключевой ставке, её изменение может влиять на процентную составляющую.'),
                        context, question='Как решение влияет на кредит?') == []


def test_related_event_cannot_prove_a_current_story(evidence):
    context = construction_context(evidence)
    context['news']['event_id'] = 'related-event'
    assert business_focus(context, event_id='current-event') is None
    assert business_focus(context, event_id='related-event')['news_ids'] == ['news']


def test_conditional_financing_analysis_can_use_product_flag_but_cannot_assert_stability(evidence):
    context = construction_context(evidence)
    card = draft('Если действующий кредит привязан к ключевой ставке, изменение ставки может влиять на процентную составляющую; наличие долга и условия договора неизвестны.')
    assert focus_issues(card, context) == []
    assert verify_draft(card, context) == []
    assert verify_draft(draft(REPORTED), context) == ['c1:unsupported_company_contract_stability']


def test_two_independently_supported_business_consequences_are_allowed(evidence):
    context = construction_context(evidence)
    card = draft('Если заказчики зависят от кредитов, дорогие кредиты могут откладывать запуск проектов.')
    card.claims[0].evidence_ids = ['news', 'industry']
    card.claims.append(card.claims[0].model_copy(update={'id': 'c2',
        'text': 'Если действующий кредит компании имеет плавающую ставку, изменение ключевой ставки может влиять на процентную составляющую.',
        'evidence_ids': ['news', 'credit']}))
    assert focus_issues(card, context) == []


@pytest.mark.parametrize('code', ['41.20', '62.01', '46.90', '49.41', '25.11'])
def test_activity_does_not_preselect_one_business_consequence(evidence, code):
    context = construction_context(evidence)
    context['industry']['value'] = code
    focus = business_focus(context)
    assert focus['channel'] == 'open_business_impact'
    assert focus['company_ids'] == ['credit', 'industry']
    assert not focus['not_inference_basis_ids']


def test_primary_event_facts_are_not_filtered_by_financial_topic_or_forecast_kind(evidence):
    context = construction_context(evidence)
    context['news']['text'] = 'Ранее Банк России сохранил ключевую ставку.'
    context['inflation'] = {'kind': 'event_fact', 'text': 'Годовая инфляция замедлилась до 7,15%.',
                            'attributes': {'topics': ['inflation']}}
    context['forecast'] = {'kind': 'event_fact', 'text': 'Эксперты ожидают снижения спроса.',
                           'attributes': {'topics': ['unlisted_business_subject'], 'kind': 'forecast'}}
    focus = business_focus(context, event_title='Инфляция в России замедлилась до 7,15%')
    assert set(focus['news_ids']) == {'inflation', 'forecast'}
    bad = draft('Если заказчики зависят от кредитов, высокая ставка может ограничивать новые заказы.')
    bad.claims[0].evidence_ids = ['news', 'industry']
    assert focus_issues(bad, context) == ['c1:business_focus_needs_primary_news_basis']


def test_reputational_credit_of_trust_is_not_financing_stability():
    from app.intelligence.financial_grounding import unsupported_financing_stability
    assert not unsupported_financing_stability('Кредит доверия клиентов может оставаться стабильным при своевременном раскрытии информации.')
