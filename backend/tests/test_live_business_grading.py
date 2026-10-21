"""Behavioral controls for curated 4/5 feeds; all data and calls are local."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from alphich_kb.personalization.grading import grade_event
from alphich_kb.personalization.inputs import normalize_event, SelectionScope
from alphich_kb.personalization.inputs import industry_matches
from alphich_kb.personalization.profiles import normalize_company
from alphich_kb.personalization.rules import match_reasons
from alphich_kb.personalization.selector import NewsSelector

FIXTURES = Path(__file__).resolve().parents[2] / "knowledge/fixtures"
AS_OF = "2026-10-21T12:00:00+00:00"


def profile(code="62"):
    raw = deepcopy(next(p for p in json.loads((FIXTURES / "company_profiles.json").read_text())["companies"]
                        if p.get("company_id", p.get("id")) == "S_SOFTWARE"))
    # Bind the selected activity using the same source feature, never an LLM's company guess.
    data = raw.get("profile", raw)
    data["industry"] = {"code": code, "name": "Контрольная деятельность"}
    data["registration_region"] = "Москва"
    raw["features"] = [{"id": "industry-proof", "company_id": "S_SOFTWARE", "name": "industry",
        "value": data["industry"], "version": raw.get("context_version", raw["profile_version"])},
        {"id": "region-proof", "company_id": "S_SOFTWARE", "name": "registration_region",
         "value": "Москва", "version": raw.get("context_version", raw["profile_version"])}]
    return raw


def event(**overrides):
    text = "На российском рынке изменились условия основной деятельности. Отдельные договоры компаний не раскрыты."
    attrs = {"topics": ["sector"], "industry_codes": ["62"], "geography": "RU",
        "kind": "report", "geographic_scope": "national", "market_scope": "sector_wide",
        "impact_scale": "material",
        "product_scope": "broad_core",
        "event_status": "reported", "business_effect": "core_market", **overrides}
    return {"id": "news", "version": 1, "company_id": None, "title": "Изменение условий рынка",
        "available_at": "2026-10-20T10:00:00+00:00", "published_at": "2026-10-20T09:00:00+00:00",
        "sources": [{"id": "source", "version": 1, "text": text, "url": "https://example.org/news",
            "published_at": "2026-10-20T09:00:00+00:00", "available_at": "2026-10-20T10:00:00+00:00"}],
        "facts": [{"id": "fact", "source_id": "source", "source_version": 1, "text": text,
            "source_span": {"start": 0, "end": len(text)}, "verification_status": "verified", "attributes": attrs}]}


def rated(raw=None, company=None):
    raw, company = raw or event(), company or profile()
    scope = SelectionScope("S_SOFTWARE", AS_OF, frozenset({raw["id"]}))
    normalized, normalized_company = normalize_event(raw), normalize_company(company)
    reasons, _ = match_reasons(normalized_company, normalized, scope)
    return grade_event(normalized_company, normalized, scope, reasons)


def test_material_main_market_event_is_4_and_carries_exact_evidence():
    result = rated()
    assert result["relevance_grade"] == 4 and result["relevance_selected"]
    assert result["relevance_assessment"]["company_feature_ids"] == ["industry-proof"]
    assert result["relevance_assessment"]["fact_ids"] == ["fact"]
    assert result["relevance_assessment"]["source_refs"][0]["source_span"]["start"] == 0


def test_linux_in_one_altai_school_is_1_for_moscow_developer():
    raw = event(geographic_scope="local", regions=["Алтайский край"], market_scope="single_case")
    raw["title"] = "В школе Алтая внедрили Linux"
    result = rated(raw)
    assert result["relevance_grade"] == 1 and not result["relevance_selected"]
    assert "operational_geography_not_confirmed" in result["relevance_assessment"]["limitations"]


def test_same_registration_city_does_not_establish_operational_presence():
    result = rated(event(geographic_scope="regional", regions=["Москва"]))
    assert result["relevance_grade"] == 3
    assert result["relevance_assessment"]["geographic_fit"] == "registration_only"


def test_big_single_project_does_not_become_relevant_by_its_amount():
    raw = event(market_scope="single_case")
    raw["title"] = "Чужая компания запустила проект за 100 миллиардов"
    assert rated(raw)["relevance_grade"] == 2


@pytest.mark.parametrize("attrs", [
    {"geographic_scope": "unknown"}, {"market_scope": "unknown"},
    {"event_status": "unknown"}, {"event_status": "proposal"}, {"event_status": "announcement"},
    {"business_effect": "unknown"}, {"regions": ["Алтайский край"]},
    {"impact_scale": "unknown"}, {"impact_scale": "limited"},
])
def test_each_missing_or_limited_dimension_prevents_grade_4(attrs):
    assert rated(event(**attrs))["relevance_grade"] <= 3


def test_new_requirement_is_5_only_when_already_in_force_for_whole_activity():
    raw = event(industry_codes=["49.4"], topics=["regulation"], event_status="in_force",
                business_effect="operating_requirement", applicability="whole_industry", effective_from="2026-09-01")
    result = rated(raw, profile("49.4"))
    assert result["relevance_grade"] == 5
    assert result["relevance_assessment"]["required_action"]
    raw["facts"][0]["attributes"]["applicability"] = "conditional"
    assert rated(raw, profile("49.4"))["relevance_grade"] == 3
    raw["facts"][0]["attributes"].update(applicability="whole_industry", effective_from="2026-11-01")
    assert rated(raw, profile("49.4"))["relevance_grade"] <= 3


def test_memory_market_forecast_is_4_for_computer_wholesaler_only():
    raw = event(industry_codes=["46.51"], geography="GLOBAL", geographic_scope="global_market",
                event_status="forecast", kind="forecast", forecast_horizon="2026-12-31")
    assert rated(raw, profile("46.51"))["relevance_grade"] == 4
    assert rated(raw, profile("49.4"))["relevance_grade"] <= 2
    raw["facts"][0]["attributes"]["forecast_horizon"] = "2026-09-30"
    assert rated(raw, profile("46.51"))["relevance_grade"] <= 3


def test_distinct_grade_five_requirements_are_not_hidden_by_topic_diversity():
    rows = []
    for index, subject in enumerate(['учёт отчётность', 'безопасность шифрование', 'лицензия разрешение',
                                     'документация подписи', 'испытания качество', 'архивирование хранение']):
        raw = event(topics=['regulation'], event_status='in_force', business_effect='operating_requirement',
                    applicability='whole_industry', effective_from='2026-09-01')
        identity = 'requirement_' + str(index)
        text = 'Обязательное требование: ' + subject + '.'
        raw.update(id=identity, title=text)
        raw['sources'][0].update(id='source_' + identity, text=text, url='https://example.org/' + identity)
        raw['facts'][0].update(id='fact_' + identity, source_id='source_' + identity, text=text,
                               source_span={'start': 0, 'end': len(text)})
        rows.append(raw)
    scope = SelectionScope('S_SOFTWARE', AS_OF, frozenset(r['id'] for r in rows))
    result = NewsSelector(rows).select(profile(), scope, mode='lexical', limit=10, min_grade=4)
    assert len(result['items']) == len(rows)
    assert all(item['relevance_grade'] == 5 for item in result['items'])
    assert result['trace']['diversity_skipped_event_ids'] == []


def test_okved_subgroup_includes_the_next_digit_without_false_neighbor_match():
    assert industry_matches("49.41", "49.4")
    assert industry_matches("49.41.2", "49.4")
    assert industry_matches("41.20.1", "41.2")
    assert not industry_matches("49.31", "49.4")
    assert not industry_matches("49", "49.4")
    assert not industry_matches("46.51", "46.1")
    assert not industry_matches("49.4", "49.41")


def test_missing_stale_or_synthetic_publication_cannot_enter_curated_feed():
    for published, synthetic in [(None, False), ("2025-01-01T00:00:00Z", False), ("2026-10-20T09:00:00Z", True)]:
        raw = event()
        raw["published_at"] = raw["sources"][0]["published_at"] = published
        raw["sources"][0]["synthetic"] = synthetic
        assert rated(raw)["relevance_grade"] <= 3


def test_one_fact_cannot_borrow_other_facts_geography_and_materiality():
    raw = event(geographic_scope="local", regions=["Москва"])
    other = deepcopy(raw["facts"][0])
    other.update(id="other-fact")
    other["attributes"].update(geographic_scope="national", regions=[], market_scope="single_case")
    raw["facts"].append(other)
    assert rated(raw)["relevance_grade"] == 3


def test_old_verified_facts_need_reviewed_business_scope_and_raw_news_cannot_fill_feed():
    raw = event()
    attrs = raw["facts"][0]["attributes"]
    for name in ["geographic_scope", "market_scope", "event_status", "business_effect"]:
        attrs.pop(name)
    assert rated(raw)["relevance_grade"] <= 3
    raw["facts"] = []
    raw["title"] = "Новость о программном обеспечении"
    scope = SelectionScope("S_SOFTWARE", AS_OF, frozenset({"news"}))
    result = NewsSelector([raw]).select(profile(), scope, min_grade=4, mode="lexical")
    assert result["items"] == []
    assert result["extraction_candidates"][0]["relevance_grade"] == 2


def test_curated_selector_never_fills_empty_slots_with_lower_grade_news():
    raw = event()
    lower = deepcopy(raw)
    lower.update(id="small", title="Отдельный отраслевой проект")
    prefix = "Единичный проект. "
    lower["sources"][0].update(id="other-source", text=prefix + lower["sources"][0]["text"])
    lower["facts"][0].update(id="other-fact", source_id="other-source", source_span={"start": len(prefix),
        "end": len(lower["sources"][0]["text"])})
    lower["facts"][0]["attributes"]["market_scope"] = "single_case"
    scope = SelectionScope("S_SOFTWARE", AS_OF, frozenset({"news", "small"}))
    result = NewsSelector([raw, lower]).select(profile(), scope, min_grade=4, mode="lexical")
    assert [i["event_id"] for i in result["items"]] == ["news"]
    assert result["items"][0]["relevance_grade"] == 4
    assert result["trace"]["below_grade_threshold"] == 1


def test_materiality_conflict_prevents_reprint_grouping_and_grade_transfer():
    first = event()
    second = deepcopy(first)
    second.update(id="different-assessment")
    second["sources"][0]["id"] = "other-source"
    second["facts"][0].update(id="other-fact", source_id="other-source")
    second["facts"][0]["attributes"]["impact_scale"] = "limited"
    scope = SelectionScope("S_SOFTWARE", AS_OF, frozenset({first["id"], second["id"]}))
    result = NewsSelector([first, second]).select(profile(), scope, min_grade=4, mode="lexical")
    assert result["items"][0]["reprint_group_size"] == 1
    assert result["trace"]["below_grade_threshold"] == 1


def test_another_sources_date_cannot_make_an_undated_fact_timely():
    raw = event()
    raw["sources"][0]["published_at"] = None
    assert rated(raw)["relevance_grade"] <= 3


def test_forecast_cannot_become_grade_5_through_contradictory_status():
    result = rated(event(kind="forecast", event_status="in_force", business_effect="operating_requirement",
                         applicability="whole_industry"))
    assert result["relevance_grade"] <= 3


def test_fresh_publication_does_not_make_last_years_observations_current():
    assert rated(event(reference_period_end="2025-12-31"))["relevance_grade"] <= 3
    assert rated(event(reference_period_end="2026-09-30"))["relevance_grade"] == 4


def test_niche_server_product_does_not_establish_wholesalers_catalogue():
    result = rated(event(industry_codes=["46.51"], product_scope="niche"), profile("46.51"))
    assert result["relevance_grade"] == 3
    assert "company_product_subsegment_not_confirmed" in result["relevance_assessment"]["limitations"]
