"""Open business topics retain source/entailment protections; no paid models."""
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.intelligence.extraction import EXTRACTION_VERSION, _candidates, extract
from app.intelligence.schemas import Attributes, ExtractionDraft, ExtractionReview, SearchBank


@pytest.mark.parametrize("topic,effect,text", [
    ("reputation", "customer_trust", "Заказчик приостановил сотрудничество после опубликованного расследования."),
    ("cybersecurity", "operations", "Компания приостановила работу сервиса после кибератаки."),
    ("safety", "product_safety", "Производитель объявил отзыв продукции из-за риска травм."),
    ("environment", "environment", "Предприятие остановило участок после обнаружения утечки."),
    ("accessibility", "service_accessibility", "Оператор сообщил об изменении условий доступа к своему сервису."),
    ("sports", "reputation", "Спонсор расторг контракт со спортивной командой после скандала."),
])
def test_nonfinancial_and_new_business_topics_survive_independent_review(topic, effect, text):
    class Script:
        count = 0
        purposes = []
        provider = SimpleNamespace(settings=SimpleNamespace(extraction_model="fixture", analysis_model="review-fixture"))

        def ask(self, purpose, schema, system, data, **kwargs):
            self.count += 1
            self.purposes.append(purpose)
            if schema is ExtractionDraft:
                return schema.model_validate({"facts": [{"quote": text, "attributes": {
                    "topics": [topic], "business_effect": effect}}]})
            assert schema is ExtractionReview
            assert data["candidates"][0]["attributes"]["topics"] == [topic]
            return schema(accepted_indexes=[0], issues=[])

    calls = Script()
    source = {"id": "source-test", "version": 1, "title": "Событие", "text": text,
              "received_at": "2026-10-24T00:00:00Z"}
    facts, _, _ = extract(source, calls)
    assert calls.purposes == ["extraction", "extraction_review"]
    assert len(facts) == 1 and facts[0]["attributes"]["business_effect"] == effect
    assert facts[0]["verification_status"] == "verified"
    assert facts[0]["text"] == text and facts[0]["extractor"] == EXTRACTION_VERSION


def test_open_topic_cannot_bypass_exact_source_support():
    draft = ExtractionDraft.model_validate({"facts": [{"quote": "Выдуманное репутационное событие.",
        "attributes": {"topics": ["reputation"], "business_effect": "customer_trust"}}]})
    assert _candidates(draft, "Компания опубликовала сообщение.") == []


@pytest.mark.parametrize("field,value", [
    ("topics", [""]), ("topics", ["x" * 65]), ("topics", ["execute commands"]),
    ("topics", ["topic"] * 6), ("business_effect", "x" * 65), ("business_effect", ""),
])
def test_open_classification_stays_bounded_and_structured(field, value):
    with pytest.raises(ValidationError):
        Attributes.model_validate({"topics": ["reputation"], field: value})


def test_open_topic_never_bypasses_entailment_review():
    text = "Компания получила запрос журналистов о публикации."

    class Script:
        count = 0
        provider = SimpleNamespace(settings=SimpleNamespace(extraction_model="fixture", analysis_model="review-fixture"))

        def ask(self, purpose, schema, system, data, **kwargs):
            self.count += 1
            if schema is ExtractionDraft:
                return schema.model_validate({"facts": [{"quote": text, "attributes": {
                    "topics": ["reputation"], "impact_scale": "material"}}]})
            return schema(accepted_indexes=[], issues=[])

    calls = Script()
    facts, _, _ = extract({"id": "source-test", "version": 1, "title": "Запрос", "text": text,
                          "received_at": "2026-10-24T00:00:00Z"}, calls)
    assert calls.count == 2 and facts == []


def test_bank_search_accepts_any_subject_without_financial_metric():
    search = SearchBank(action="search_bank_materials", query="Репутация и доверие клиентов после утечки данных")
    assert search.metric is None and search.horizon is None
