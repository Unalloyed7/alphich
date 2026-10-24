"""Offline verifier protocol, scope, shared budgets and bounded exact-source reads."""
from copy import deepcopy
from dataclasses import replace
from decimal import Decimal
import json

import httpx
import pytest
from sqlalchemy import select

from alphich_kb.personalization.inputs import SelectionScope
from app.core import Base, make_session_factory
from app.intelligence.agent import REVIEW, Tools
from app.intelligence.provider import Calls, Provider, canonical, input_bytes
from app.intelligence.schemas import CardDraft, FinalVerificationDecision, SemanticReview, VerificationDecision
from app.intelligence.settings import AnalysisError, Model, SchemaFailure, Settings
from app.intelligence.storage import BudgetDay, LLMCall
from app.intelligence.verifier import INSPECTION_LIMIT_BYTES, VerificationAgent


ORIGINAL = "Исходный обзор банка. " + "Проверяем исходный текст. " * 85 + " Конец первоисточника."


@pytest.fixture
def context(tmp_path):
    engine, factory = make_session_factory(f"sqlite:///{tmp_path / 'verifier.db'}")
    Base.metadata.create_all(engine)
    model = Model("verifier-unit-model", "qwen3.6-35b-a3b", Decimal("1"), Decimal("2"), input_limit=65536)
    settings = Settings(base_url="https://gateway.example.invalid/v1", api_key="unit-verifier-secret",
                        extraction_model=model.id, analysis_model=model.id, models={model.id: model}, prices_confirmed=True)
    scope = SelectionScope("COMPANY_A", "2026-10-21T00:00:00Z", frozenset({"allowed-event"}), "cfo")
    tools = Tools({"id": "COMPANY_A", "features": []}, {"id": "allowed-event", "facts": []}, {}, scope)
    tools.registry.update({
        "bank-original": {"id": "bank-original", "kind": "document_chunk", "document_id": "bank-allowed", "text": ORIGINAL},
        "bank-reviewed": {"id": "bank-reviewed", "kind": "bank_claim", "document_id": "bank-allowed",
                          "document_chunk_id": "bank-original", "text": "Исходный обзор банка.", "position_role": "current_in_document"}})
    draft = CardDraft.model_validate({
        "claims": [{"id": "c1", "kind": "bank_view", "text": "Исходный обзор банка.", "evidence_ids": ["bank-reviewed"]}],
        "next_steps": [{"text": "Уточнить применимые условия", "basis_claim_ids": ["c1"], "conditions": ["Если есть договор"]}]})
    yield factory, settings, tools, draft
    engine.dispose()


@pytest.fixture
def scripted_calls(context):
    factory, settings, _, _ = context
    clients = []

    def build(responses, *, count=0, configured=None):
        seen = []

        def gateway(request):
            payload = json.loads(request.content)
            schema = json.loads(payload["messages"][0]["content"].split("schema:\n", 1)[1])
            data = json.loads(payload["messages"][1]["content"])["untrusted_data"]
            seen.append({"schema": schema, "data": data, "system": payload["messages"][0]["content"]})
            assert len(seen) <= len(responses), "Unexpected paid follow-up"
            response = responses[len(seen) - 1]
            if isinstance(response, Exception):
                raise response
            result = response(data) if callable(response) else response
            return httpx.Response(200, json={"choices": [{"message": {"content": canonical(result)}, "finish_reason": "stop"}],
                                            "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

        client = httpx.Client(transport=httpx.MockTransport(gateway))
        clients.append(client)
        calls = Calls(Provider(factory, configured or settings, client), "verifier-run")
        calls.count = count
        return calls, seen

    yield build
    for client in clients:
        client.close()


def verdict(**changes):
    return {"decision": {"action": "verdict", "supported_claim_ids": ["c1"],
                         "supported_action_indexes": [0], "issues": [], **changes}}


def inspect_original(data):
    original = next(item for item in data["evidence"] if item["kind"] == "document_chunk")
    assert original["prompt_excerpt_only"] is True and original["text"] != ORIGINAL
    return {"decision": {"action": "inspect_evidence", "ids": [original["id"]]}}


def test_verifier_reads_original_evidence_then_returns_verdict(context, scripted_calls):
    _, _, tools, draft = context
    saved = deepcopy(tools.registry)

    def after_read(data):
        assert data["inspection_allowed"] is False
        assert data["question"] == "Подтверждён ли тезис источником?"
        source = data["inspection"]["evidence"][0]
        assert source["text"] == ORIGINAL
        assert "Конец первоисточника." in source["text"]
        return verdict()

    calls, seen = scripted_calls([inspect_original, after_read])
    review = VerificationAgent(tools, calls, "EXTRA REVIEW RULE").review(
        draft, question="Подтверждён ли тезис источником?", round_index=2, repaired=False)
    assert isinstance(review, SemanticReview) and review.supported_claim_ids == ["c1"]
    assert calls.count == 2
    assert [step["schema"]["title"] for step in seen] == ["VerificationDecision", "FinalVerificationDecision"]
    assert "EXTRA REVIEW RULE" in seen[0]["system"]
    assert tools.registry == saved
    assert tools.tool_count == 1
    assert tools.steps[-1]["agent_role"] == "verifier"
    assert tools.steps[-1]["action"] == "get_evidence"
    assert tools.steps[-1]["arguments"]["ids"] == ["bank-original"]


@pytest.mark.parametrize("count,repaired,tools_used", [(5, False, 0), (7, True, 0), (0, False, 8)])
def test_call_or_tool_cap_forces_a_final_verdict(context, scripted_calls, count, repaired, tools_used):
    _, _, tools, draft = context
    tools.steps.extend({"action": "get_evidence", "origin": "agent_tool"} for _ in range(tools_used))
    calls, seen = scripted_calls([verdict()], count=count)
    VerificationAgent(tools, calls, "Review").review(draft, question=None, round_index=0, repaired=repaired)
    assert len(seen) == 1 and seen[0]["schema"]["title"] == "FinalVerificationDecision"
    assert seen[0]["data"]["inspection_allowed"] is False
    assert tools.tool_count == tools_used


def test_initial_inspection_keeps_two_calls_for_repair_and_reverification(context, scripted_calls):
    _, _, tools, draft = context
    calls, seen = scripted_calls([inspect_original, verdict()], count=4)
    VerificationAgent(tools, calls, "Review").review(draft, question=None, round_index=0, repaired=False)
    assert calls.count == 6 and len(seen) == 2


def test_verifier_cannot_inspect_a_foreign_id_or_follow_up(context, scripted_calls):
    _, _, tools, draft = context
    calls, seen = scripted_calls([{"decision": {"action": "inspect_evidence", "ids": ["foreign-company-evidence"]}}])
    with pytest.raises(AnalysisError, match="evidence_outside_scope"):
        VerificationAgent(tools, calls, "Review").review(draft, question=None, round_index=0, repaired=False)
    assert len(seen) == 1
    assert tools.steps[-1]["result"] == {"error": "evidence_outside_scope"}


@pytest.mark.parametrize("changes", [{"supported_claim_ids": ["c99"]}, {"supported_claim_ids": ["c1", "c1"]},
                                     {"supported_action_indexes": [1]}, {"supported_action_indexes": [-1]},
                                     {"supported_action_indexes": [0, 0]}])
def test_invalid_verdict_references_fail_without_another_call(context, scripted_calls, changes):
    _, _, tools, draft = context
    calls, seen = scripted_calls([verdict(**changes)])
    with pytest.raises(AnalysisError, match="invalid_verification_review"):
        VerificationAgent(tools, calls, "Review").review(draft, question=None, round_index=0, repaired=False)
    assert len(seen) == 1


def test_inspection_is_once_only_even_if_model_requests_another(context, scripted_calls):
    _, _, tools, draft = context
    calls, seen = scripted_calls([inspect_original,
                                 {"decision": {"action": "inspect_evidence", "ids": ["bank-original"]}}])
    with pytest.raises(SchemaFailure):
        VerificationAgent(tools, calls, "Review").review(draft, question=None, round_index=0, repaired=False)
    assert len(seen) == 2 and tools.tool_count == 1


def test_large_originals_have_bounded_observation_and_remain_immutable(context, scripted_calls):
    _, _, tools, draft = context
    for index in range(3):
        tools.registry[f"large-{index}"] = {"id": f"large-{index}", "kind": "document_chunk",
                                            "text": '\\"Большой исходный текст. ' * 2000,
                                            "metadata": {"payload": "nested" * 5000}}
    saved = deepcopy(tools.registry)

    def after_read(data):
        inspection = data["inspection"]
        assert len(canonical(inspection).encode()) <= INSPECTION_LIMIT_BYTES
        assert inspection["truncated"] is True
        assert len(inspection["evidence"]) == 3
        assert all(item["inspection_excerpt_only"] and item["metadata_omitted"] for item in inspection["evidence"])
        assert all(item["text"] for item in inspection["evidence"])
        return verdict(issues=["unsupported_claim"], supported_claim_ids=[])

    calls, _ = scripted_calls([{"decision": {"action": "inspect_evidence", "ids": ["large-0", "large-1", "large-2"]}}, after_read])
    review = VerificationAgent(tools, calls, "Review").review(draft, question=None, round_index=0, repaired=False)
    assert review.issues == ["unsupported_claim"]
    assert tools.registry == saved


def test_verifier_uses_same_calls_run_ledger_and_budget(context, scripted_calls):
    factory, settings, tools, draft = context
    prior = {key: value for key, value in verdict()["decision"].items() if key != "action"}
    calls, seen = scripted_calls([prior, inspect_original, verdict()],
                                configured=replace(settings, max_calls=3))
    # A prior agent already used this exact Calls object and shared run budget.
    calls.ask("prior_agent", SemanticReview, "Prior review", {}, extraction=False)
    VerificationAgent(tools, calls, "Review").review(draft, question=None, round_index=5, repaired=True)
    assert calls.count == 3
    with pytest.raises(AnalysisError, match="agent_call_limit"):
        calls.ask("forbidden-fourth", SemanticReview, "Review", {})
    assert len(seen) == 3
    with factory() as db:
        records = db.scalars(select(LLMCall)).all()
        assert {row.run_id for row in records} == {"verifier-run"}
        assert {row.purpose for row in records} == {"prior_agent", "semantic_review_5", "semantic_review_5_final"}
        budget = db.scalar(select(BudgetDay))
        assert budget.spent_micros == 60 and budget.reserved_micros == 0


def test_unknown_paid_outcome_is_not_retried(context, scripted_calls):
    factory, _, tools, draft = context
    calls, seen = scripted_calls([httpx.ReadTimeout("fixture upstream failure")])
    verifier = VerificationAgent(tools, calls, "Review")
    with pytest.raises(AnalysisError, match="llm_timeout_outcome_unknown"):
        verifier.review(draft, question=None, round_index=0, repaired=False)
    restarted = VerificationAgent(tools, Calls(calls.provider, calls.run_id), "Review")
    with pytest.raises(AnalysisError, match="previous_call_outcome_unknown"):
        restarted.review(draft, question=None, round_index=0, repaired=False)
    assert len(seen) == 1
    with factory() as db:
        budget = db.scalar(select(BudgetDay))
        assert budget.spent_micros == 0 and budget.reserved_micros > 0


@pytest.mark.parametrize("response_mode", ["text", "json_object", "json_schema"])
def test_dense_context_fits_real_provider_without_trimming_draft_or_question(context, response_mode):
    factory, settings, tools, _ = context
    settings = replace(settings, models={key: replace(value, input_limit=32768, response_mode=response_mode,
                                                     reasoning_enabled=False)
                                         for key, value in settings.models.items()})
    # Many excerpts plus a large valid draft exceed the original verifier input.
    for index in range(20):
        tools.registry[f"dense-{index}"] = {"id": f"dense-{index}", "kind": "document_chunk",
                                            "document_id": "bank-allowed", "text": 'Контекст \\" ' * 190}
    question = "Проверить поддержку каждого тезиса источниками? " * 12
    model = settings.models[settings.analysis_model]

    def sized_draft(repetitions):
        return CardDraft.model_validate({"claims": [
            {"id": f"c{index + 1}", "kind": "bank_view", "text": "Тезис обзора. " * repetitions,
             "evidence_ids": ["bank-reviewed"]} for index in range(10)]})

    # Keep required input inside the current prompt/schema budget. A fixed-size
    # draft can become an oversized-required-input case when safeguards grow;
    # that separate fail-before-dispatch behavior is tested below. The dense
    # sources still overflow and must be bounded without altering this draft or
    # the full question in either verifier call.
    probe = VerificationAgent(tools, None, REVIEW)
    minimum = {"company_id": tools.scope.company_id, "role": tools.scope.role, "as_of": tools.scope.as_of,
               "question": question, "draft": tools.expose(sized_draft(1).model_dump()), "evidence": [],
               "inspection_allowed": True, "remaining_calls": 8,
               "context_truncated": True, "context_limitations": ["verifier_context_truncated"]}
    overhead = input_bytes(model, probe.system, minimum, VerificationDecision.model_json_schema())
    repetitions = min(70, 1 + (model.input_limit - overhead - 2000) // (10 * len("Тезис обзора. ".encode())))
    assert repetitions > 1
    draft = sized_draft(repetitions)
    saved = deepcopy(tools.registry)
    seen = []
    expected_draft = tools.expose(draft.model_dump())

    def gateway(request):
        payload = json.loads(request.content)
        assert len(canonical(payload).encode()) + 256 <= model.input_limit
        data = json.loads(payload["messages"][1]["content"])["untrusted_data"]
        assert data["draft"] == expected_draft and data["question"] == question
        seen.append(data)
        if len(seen) == 1:
            assert data["context_truncated"] is True
            original = next(item for item in data["evidence"] if item["id"] == tools.aliases["bank-reviewed"])
            result = {"decision": {"action": "inspect_evidence", "ids": [original["id"]]}}
        else:
            inspected = data["inspection"]["evidence"]
            assert inspected[0]["text"] == tools.registry["bank-reviewed"]["text"]
            assert all(item["id"] != inspected[0]["id"] for item in data["evidence"])
            result = verdict(supported_claim_ids=[claim.id for claim in draft.claims], supported_action_indexes=[])
        return httpx.Response(200, json={"choices": [{"message": {"content": canonical(result)}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        calls = Calls(Provider(factory, settings, client), "dense-verifier-run")
        verifier = VerificationAgent(tools, calls, REVIEW)
        unbounded = {"company_id": tools.scope.company_id, "role": tools.scope.role, "as_of": tools.scope.as_of,
                     "question": question, "draft": expected_draft, "evidence": tools.prompt_evidence(),
                     "inspection_allowed": True, "remaining_calls": 8}
        assert input_bytes(model, verifier.system, unbounded, VerificationDecision.model_json_schema()) > model.input_limit
        with pytest.raises(AnalysisError, match="context_too_large"):
            calls.provider.generate("dense-proof", "unbounded-control", model.id, verifier.system,
                                    unbounded, VerificationDecision.model_json_schema())
        assert seen == []
        review = verifier.review(draft, question=question, round_index=0, repaired=False)
    assert review.supported_claim_ids == [claim.id for claim in draft.claims]
    assert len(seen) == 2 and calls.count == 2
    assert tools.registry == saved


def test_oversized_required_draft_fails_before_any_paid_call(context, scripted_calls):
    _, settings, tools, _ = context
    configured = replace(settings, models={key: replace(value, input_limit=4096) for key, value in settings.models.items()})
    draft = CardDraft.model_validate({"claims": [
        {"id": f"c{index + 1}", "kind": "bank_view", "text": "Большой обязательный тезис. " * 35,
         "evidence_ids": ["bank-reviewed"]} for index in range(12)]})
    calls, seen = scripted_calls([], configured=configured)
    with pytest.raises(AnalysisError, match="context_too_large"):
        VerificationAgent(tools, calls, REVIEW).review(draft, question="Полный вопрос без обрезки", round_index=0, repaired=False)
    assert calls.count == 0 and seen == []


def test_inspection_is_reduced_when_required_draft_uses_most_input_budget(context):
    factory, settings, tools, _ = context
    settings = replace(settings, models={key: replace(value, input_limit=32768) for key, value in settings.models.items()})
    # Size the mandatory draft against the current prompt/schema, keeping room
    # for source identities but not a full original. Prompt wording must not
    # accidentally turn this into the separate oversized-required-input case.
    probe_calls = Calls(Provider(factory, settings), "size-only-no-http")
    probe = VerificationAgent(tools, probe_calls, REVIEW)
    skeleton = {"company_id": tools.scope.company_id, "role": tools.scope.role, "as_of": tools.scope.as_of,
                "question": None, "draft": {"claims": [
                    {"id": f"c{i + 1}", "kind": "bank_view", "text": "Я" * 6,
                     "evidence_ids": [tools.expose("bank-reviewed")], "conditions": []} for i in range(12)],
                    "risks": [], "opportunities": [], "next_steps": [], "missing_data": []},
                "evidence": [], "inspection_allowed": True, "remaining_calls": 8,
                "context_truncated": True, "context_limitations": ["verifier_context_truncated"]}
    overhead = input_bytes(settings.models[settings.analysis_model], probe.system, skeleton,
                           VerificationDecision.model_json_schema())
    text_chars = 6 + (32768 - overhead - 1200) // (12 * len("Я".encode()))
    assert 6 < text_chars <= 1000
    draft = CardDraft.model_validate({"claims": [
        {"id": f"c{index + 1}", "kind": "bank_view", "text": "Я" * text_chars,
         "evidence_ids": ["bank-reviewed"]} for index in range(12)]})
    saved = deepcopy(tools.registry)
    expected_draft = tools.expose(draft.model_dump())
    seen = []

    def gateway(request):
        payload = json.loads(request.content)
        assert len(canonical(payload).encode()) + 256 <= 32768
        data = json.loads(payload["messages"][1]["content"])["untrusted_data"]
        assert data["draft"] == expected_draft
        seen.append(data)
        if len(seen) == 1:
            source = next(item for item in data["evidence"] if item["id"] == tools.aliases["bank-original"])
            result = {"decision": {"action": "inspect_evidence", "ids": [source["id"]]}}
        else:
            source = data["inspection"]["evidence"][0]
            assert source["text"] and source["text"] != ORIGINAL
            assert source["inspection_excerpt_only"] is True
            assert data["inspection"]["truncated"] is True and data["context_truncated"] is True
            assert not any(item["id"] == source["id"] for item in data["evidence"])
            result = verdict(supported_claim_ids=[], supported_action_indexes=[], issues=["unsupported_claim"])
        return httpx.Response(200, json={"choices": [{"message": {"content": canonical(result)}, "finish_reason": "stop"}],
                                        "usage": {"prompt_tokens": 10, "completion_tokens": 5}})

    with httpx.Client(transport=httpx.MockTransport(gateway)) as client:
        calls = Calls(Provider(factory, settings, client), "tight-verifier-run")
        verifier = VerificationAgent(tools, calls, REVIEW)
        naive_followup = {"company_id": tools.scope.company_id, "role": tools.scope.role, "as_of": tools.scope.as_of,
                          "question": None, "draft": expected_draft, "evidence": tools.prompt_evidence(),
                          "inspection_allowed": False, "remaining_calls": 7,
                          "inspection": {"evidence": [tools.expose(tools.registry["bank-original"])]}}
        assert input_bytes(settings.models[settings.analysis_model], verifier.system, naive_followup,
                           FinalVerificationDecision.model_json_schema()) > 32768
        review = verifier.review(draft, question=None, round_index=0, repaired=False)
    assert review.issues == ["unsupported_claim"] and len(seen) == 2
    assert tools.registry == saved
