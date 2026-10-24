"""A graded news item must survive company-tool reads in the real wire budget."""
from copy import deepcopy
from decimal import Decimal
from types import SimpleNamespace

from alphich_kb.personalization.inputs import SelectionScope
from app.intelligence.agent import SYSTEM, Tools, analyze
from app.intelligence.financial_grounding import BUSINESS_RULES, business_focus
from app.intelligence.provider import input_bytes
from app.intelligence.schemas import Decision
from app.intelligence.settings import Model, Settings


def test_company_tool_reads_do_not_repeat_grade_diagnostics_or_exhaust_input():
    values = {"okved_code": "41.20", "okved_name": "Строительство", "region_registration": "Москва",
              "revenue_raw": "20000000000", "revenue_period": "2025", "revenue_currency": "RUB",
              "revenue_unit": "unknown", "revenue_scope": "unknown", "CL_PRODUCT_LOA_MARK": True,
              "fts_flag": True, "fts_countries": ["Китай"]}
    profile = {**values, "id": "own", "profile_version": "v1", "products": {"CL_PRODUCT_LOA_MARK": True}, "industry": {
        "okved_code": "41.20", "okved_name": "Строительство"}, "features": [
            {"id": name, "name": name, "value": value, "profile_version": "v1",
             "source_refs": ["organizer.csv:row:2", "organizer-webinar:source-clarification"]}
            for name, value in values.items()]}
    texts = ["Изменён порядок оценки заявок на участие в государственных и муниципальных закупках.",
             "Заказчики вправе учитывать деловую репутацию участников строительных закупок.",
             "Для оценки квалификации применяются критерии, указанные в документации закупки.",
             "Изменения применяются к предусмотренным постановлением конкурсным процедурам."]
    event = {"id": "news", "title": "В строительных закупках учитывается деловая репутация подрядчиков",
             "published_at": None, "facts": [{"id": f"news-{index}", "text": text}
                                            for index, text in enumerate(texts)]}
    reason = ("Если компания участвует в применимых закупках, изменение правил оценки деловой репутации "
              "может потребовать проверки состава подтверждающих документов. Участие неизвестно. ") * 4
    gate = {"method": "company-context-bidirectional-cosine-v3-open-business", "similarity": 0.79,
            "profile_hash": "a" * 64, "event_hash": "b" * 64, "company_context_hash": "c" * 64,
            "facet_similarities": {name: 0.8 for name in values},
            "company_context_feature_ids": list(values), "limitations": [
                "semantic_match_does_not_prove_business_exposure", "requires_independent_threshold_validation"]}
    selection = {"relevance_grade": 4, "relevance_selected": True, "relevance_assessment": {
        "semantic_gate": gate, "business_channel": reason,
        "judgment": {"grade": 4, "reason": reason, "company_feature_ids": ["okved_code"],
                     "news_fact_ids": ["news-0"], "missing_data": ["Участие компании в закупках неизвестно"],
                     "required_action": "Проверить применимость требований к закупкам компании."}},
        "reasons": [{"code": "glm_business_relevance", "explanation": reason}]}
    model = Model("unit-business-context", "glm-5.3-flash", Decimal("1"), Decimal("2"),
                  input_limit=32768, output_limit=2500, response_mode="json_schema", reasoning_enabled=False)
    settings = Settings(base_url="https://gateway.example.invalid/v1", api_key="unit-no-http",
                        extraction_model=model.id, analysis_model=model.id, models={model.id: model})
    scope = SelectionScope("own", "2026-10-24T13:49:09Z", frozenset({"news"}), "cfo")
    tools = Tools(profile, event, selection, scope)
    saved_registry, saved_selection = deepcopy(tools.registry), deepcopy(selection)

    # The production failure occurred after a company-tool response had been
    # repeated alongside full E5/judge diagnostics and the verbose focus policy.
    full_tool_reads = [{"evidence": list(tools.registry.values())[start:start + 5], "limitations": []}
                       for start in (0, 5)]
    old_control = {"company_id": scope.company_id, "role": scope.role, "as_of": scope.as_of,
                   "event": {key: event.get(key) for key in ("id", "title", "published_at")}, "question": None,
                   "business_focus": tools.expose(business_focus(tools.registry, event_id="news")),
                   "business_relevance": tools.expose(selection), "relevance_reasons": tools.expose(selection["reasons"]),
                   "observations": tools.expose(full_tool_reads), "evidence": [], "must_finish": False}
    assert input_bytes(model, SYSTEM + BUSINESS_RULES, old_control, Decision.model_json_schema()) > model.input_limit

    class LocalCalls:
        def __init__(self):
            self.provider = SimpleNamespace(settings=settings)
            self.count, self.seen = 3, []  # Extraction and grading already ran.

        def heartbeat(self):
            pass

        def ask(self, purpose, schema, system, data):
            assert input_bytes(model, system, data, schema.model_json_schema()) <= model.input_limit
            self.seen.append((purpose, deepcopy(data)))
            self.count += 1
            assert {item["kind"] for item in data["evidence"]} >= {"event_fact", "company_feature"}
            if purpose in {"agent_0", "agent_1"}:
                start = 0 if purpose == "agent_0" else 5
                result = {"decision": {"action": "get_company_features",
                    "ids": data["inference_basis"]["company_ids"][start:start + 5]}}
            elif purpose.startswith("agent_"):
                news, company = data["inference_basis"]["news_ids"][0], data["inference_basis"]["company_ids"][0]
                result = {"decision": {"action": "finish", "draft": {
                    "fact_evidence_ids": [news, company], "inferences": [{"id": "i1",
                        "text": "Если компания участвует в закупках, оценка деловой репутации может изменить требования к подтверждению квалификации.",
                        "evidence_ids": [news, company], "conditions": ["Участие компании в закупках неизвестно"]}],
                    "next_steps": [{"text": "Тендерному отделу проверить применимые закупки и при необходимости состав подтверждающих документов.",
                        "basis_inference_ids": ["i1"], "conditions": ["Есть применимые закупки"]}],
                    "missing_data": ["Участие компании в закупках неизвестно"]}}}
            else:
                assert "Участие компании в закупках неизвестно" in data["business_relevance"]["missing_data"]
                result = {"decision": {"action": "verdict", "issues": [], "supported_action_indexes": [0],
                    "supported_claim_ids": [claim["id"] for claim in data["draft"]["claims"]]}}
            return schema.model_validate(result)

    calls = LocalCalls()
    card, verification = analyze(tools, calls)
    assert verification["status"] == "passed_with_conditions" and card["next_steps"]
    assert [purpose for purpose, _ in calls.seen] == ["agent_0", "agent_1", "agent_2", "semantic_review_2"]
    for _, data in calls.seen:
        assert "semantic_gate" not in data["business_relevance"]
        assert data["business_focus"]["selection_policy"]["impact_domains_open"] is True
        assert data["business_focus"]["news_ids"]
        assert "hypothesis" not in data["business_focus"]
        for observation in data.get("observations", []):
            if "evidence" in observation:
                assert observation["model_context_summary"] is True
                assert all(item["text"] and item["original_evidence_retained"] for item in observation["evidence"])
                assert all("source_refs" not in item for item in observation["evidence"])
    assert tools.registry == saved_registry and selection == saved_selection
    raw_reads = [step["result"] for step in tools.steps if step["action"] == "get_company_features"]
    assert len(raw_reads) == 2 and all("source_refs" in item for result in raw_reads for item in result["evidence"])
