"""Business evidence and administrative labels have different permissions."""
import copy
import json
from pathlib import Path

from alphich_kb.personalization.business import business_context
from alphich_kb.personalization.facets import build_facets
from alphich_kb.personalization.inputs import SelectionScope
from alphich_kb.personalization.profiles import normalize_company
from alphich_kb.personalization.selector import NewsSelector
from app.intelligence.agent import Tools
from app.intelligence.evidence import company_evidence, verify_draft
from app.intelligence.schemas import CardDraft


def profile(extra=None, products=None):
    values = {"okved_code": "62.01", "okved_name": "Разработка программного обеспечения",
              **(extra or {})}
    features = [{"id": name, "name": name, "value": value, "profile_version": "v1"}
                for name, value in {**values, **(products or {})}.items()]
    return {"id": "own", "profile_version": "v1", "industry": {
        "okved_code": values["okved_code"], "okved_name": values["okved_name"]},
        "products": products or {}, "features": features}


def test_staff_and_guarantee_reach_search_context_and_decisions_without_claiming_exposure():
    p = profile({"employees_band": "101-250", "employees_fns_band": "101-250",
                 "business_stage_label": "Оптимизация", "client_type": "CLIENT"},
                {"CL_PRODUCT_GAR_MARK": True})
    before = copy.deepcopy(p)
    context = business_context(p)
    assert p == before
    assert {h["id"] for h in context["hypotheses"]} >= {"workforce_costs", "contract_security"}
    assert all(h["status"] == "requires_confirmation" for h in context["hypotheses"])
    observed = {row["code"]: row for row in context["observed"]}
    assert "неизвестны" in observed["employees_fns_band"]["meaning"]
    assert "методика и дата неизвестны" in observed["business_stage_label"]["meaning"]
    facets = build_facets(normalize_company(p), "cfo")
    assert next(f for f in facets if f["kind"] == "employment")["feature_ids"] == [
        "employees_band", "okved_code", "okved_name"]
    assert "гарантий" in next(f for f in facets if f["kind"] == "finance")["text"]
    tools = Tools(p, {"id": "news", "facts": []}, {},
                  SelectionScope("own", "2026-10-23T00:00:00Z", frozenset({"news"})))
    prompt = tools.prompt_business_context({"driver": "labor", "not_inference_basis_ids": []})
    # A labor hint must not hide other supported context or preselect the only
    # permitted business mechanism for the author/reviewer.
    assert {h["id"] for h in prompt["hypotheses"]} >= {
        "delivery_economics", "workforce_costs", "contract_security", "customer_it_budgets"}
    assert prompt["scope"] == "optional_examples_not_a_closed_catalog_of_impacts"
    assert any("labor" in check["topics"] for check in prompt["factor_checks"])
    assert not any(check["factor"] == "client_type" for check in prompt["factor_checks"])


def test_disagreeing_staff_sources_do_not_become_a_workforce_estimate():
    p = profile({"employees_fns_band": "101-250", "employees_spark_band": "1-10"})
    context = business_context(p)
    observed = {row["code"]: row for row in context["observed"]}
    assert observed["employees_fns_band"]["value"] == "101-250"
    assert observed["employees_spark_band"]["value"] == "1-10"
    assert "employees_band" not in observed
    assert "workforce_costs" not in {h["id"] for h in context["hypotheses"]}
    assert "employment" not in {f["kind"] for f in build_facets(normalize_company(p), "cfo")}


def test_opaque_labels_alone_cannot_substantiate_a_business_inference():
    p = profile({"business_stage_label": "Оптимизация", "negative_mark_raw": "1", "client_type": "CLIENT"})
    evidence = company_evidence(p)
    evidence["news"] = {"kind": "event_fact", "text": "Рынок изменился."}
    assert "Не является подтверждением банкротства" in evidence["negative_mark_raw"]["text"]
    draft = CardDraft.model_validate({"claims": [{"id": "c1", "kind": "model_inference",
        "text": "Если изменения существенны, компания может пересмотреть план.",
        "evidence_ids": ["news", "business_stage_label", "client_type"],
        "conditions": ["Влияние требуется подтвердить"]}]})
    assert "c1:administrative_labels_do_not_establish_business_impact" in verify_draft(draft, evidence)


def test_stage_ownership_and_staff_do_not_override_the_existing_relevance_grade():
    events = json.loads((Path(__file__).resolve().parents[2] / "knowledge/fixtures/news_events.json").read_text())["events"]
    event = next(row for row in events if row["id"] == "software_update")
    selector = NewsSelector([event])
    scope = SelectionScope("own", "2026-10-20T12:00:00Z", frozenset({event["id"]}))
    original = selector.select(profile(), scope, mode="lexical")
    enriched = selector.select(profile({"company_age_band": "до года", "business_stage_label": "Рост",
        "negative_mark_raw": "1", "ownership_form": "ГОСУДАРСТВЕННАЯ", "employees_band": "1000+"}),
        scope, mode="lexical")
    assert len(original["items"]) == len(enriched["items"]) == 1
    assert original["items"][0]["relevance_grade"] == enriched["items"][0]["relevance_grade"]
    assert enriched["items"][0]["relevance_grade"] < 4
