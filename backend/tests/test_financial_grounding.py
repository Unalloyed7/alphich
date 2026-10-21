"""A forecast and a credit flag cannot establish the path of a company's loan terms."""
from copy import deepcopy

import pytest

from app.intelligence.evidence import verify_draft
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
