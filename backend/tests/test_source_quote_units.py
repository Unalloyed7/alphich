"""The model selects IDs; every published quote is still an exact original span."""
from app.intelligence.extraction import quote_units, _candidates
from app.intelligence.schemas import ExtractionDraft


def draft(quote_id, **changes):
    return ExtractionDraft.model_validate({"facts": [{"quote_id": quote_id,
        "attributes": {"topics": ["regulation"], **changes}}]})


def test_quote_id_materializes_original_whitespace_and_coordinates():
    text = "Первое предложение.\nТребование вступило в силу.  Прочие условия."
    windows = [{"start": 0, "end": len(text)}]
    units = quote_units(text, windows)
    chosen = next(u for u in units if "вступило" in u["text"])
    result = _candidates(draft(chosen["id"]), text, windows, text, units)
    assert len(result) == 1
    span = result[0]["source_span"]
    assert result[0]["text"] == text[span["start"]:span["end"]] == chosen["text"]


def test_nonexistent_unit_is_rejected_and_model_text_cannot_replace_server_quote():
    text = "Подтверждённое сообщение."
    windows = [{"start": 0, "end": len(text)}]
    units = quote_units(text, windows)
    assert _candidates(draft("q999"), text, windows, text, units) == []
    bad = draft("q1")
    bad.facts[0].quote = "Сочинённый текст источника."
    result = _candidates(bad, text, windows, text, units)
    assert result[0]["text"] == text
    assert "Сочинённый" not in result[0]["text"]


def test_units_never_cross_hidden_windows_or_resolve_unseen_content():
    text = "Начало сообщения. СКРЫТЫЙ ПРОПУСК. Конец сообщения."
    windows = [{"start": 0, "end": 17}, {"start": 35, "end": len(text)}]
    units = quote_units(text, windows)
    for unit in units:
        assert any(w["start"] <= unit["start"] < unit["end"] <= w["end"] for w in windows)
        assert text[unit["start"]:unit["end"]] == unit["text"]
        assert "СКРЫТЫЙ" not in unit["text"]


def test_rate_decision_cannot_turn_into_an_operating_obligation():
    text = "Банк России сохранил ключевую ставку."
    units = quote_units(text, [{"start": 0, "end": len(text)}])
    result = _candidates(draft("q1", kind="rate_decision", business_effect="operating_requirement", event_status="in_force"),
        text, context=text, units=units)
    assert result[0]["attributes"]["business_effect"] == "financing"
    assert result[0]["attributes"]["event_status"] == "reported"
