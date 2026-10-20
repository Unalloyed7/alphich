"""Real specialist decisions and shared ledger with a scripted local gateway."""
import copy
import json

import httpx
import pytest
from sqlalchemy import select

from alphich_kb.personalization.inputs import SelectionScope
from app.intelligence.agent import Tools
from app.intelligence.provider import Calls, Provider
from app.intelligence.researcher import BankResearchAgent
from app.intelligence.schemas import ExtractionReview, SearchBank
from app.intelligence.settings import AnalysisError, SchemaFailure
from app.intelligence.storage import LLMCall
from test_intelligence import store  # Shared isolated DB/settings fixture; no live gateway.


CURRENT = {"id": "bank-current", "document_id": "allowed-document", "document_chunk_id": "chunk-current",
           "kind": "forecast", "text": "Банк ожидает сохранения условий на конец года.", "metric": "policy_rate",
           "forecast_horizon": "2026-12-31", "geography": "RU", "scenario": "base", "attributed_to": "alfa_bank",
           "position_role": "current_in_document", "verification_status": "visually_verified", "published_at": "2026-09-11"}


class ScopedBank:
    """Deterministic retriever stub that actually applies the received access scope."""
    def __init__(self, claims=None):
        self.claims = claims if claims is not None else [copy.deepcopy(CURRENT)]
        self.requests = []

    def retrieve(self, query, scope, **kwargs):
        self.requests.append((query, scope, kwargs))
        claims = [copy.deepcopy(c) for c in self.claims if scope.permits(c)]
        evidence = [{"id": c["document_chunk_id"], "kind": "document_chunk", "document_id": c["document_id"],
                     "text": c["text"], "page": 1, "published_at": c["published_at"],
                     "source_refs": [{"document_id": c["document_id"], "page": 1}]} for c in claims]
        return {"retrieval_run_id": "stub-retrieval", "trace": {"as_of": scope.as_of, "corpus_version": "test-v1",
                "used_context_chars": sum(len(e["text"]) for e in evidence), "max_context_chars": kwargs["max_context_chars"]},
                "status": "ok" if claims else "not_found", "bank_view_status": "found" if claims else "not_found",
                "bank_claims": claims, "evidence": evidence, "parent_contexts": [], "limitations": []}


def make_tools(bank=None, *, as_of="2026-10-21T00:30:00+03:00"):
    scope = SelectionScope("DEMO_A", as_of, frozenset({"event"}), "cfo")
    event = {"id": "event", "title": "Учебное решение регулятора", "published_at": "2026-09-11T10:30:00+00:00",
             "facts": [{"id": "news-fact", "text": "Банк России сохранил ключевую ставку.",
                        "source_available_at": "2026-10-20T12:00:00+00:00"}]}
    return Tools({"id": "DEMO_A", "features": []}, event, {"bank_questions": []}, scope,
                 (bank, frozenset({"allowed-document"})) if bank else None)


class ResearchGateway:
    def __init__(self, *, search_first=True, handoff=None, extra_search_fields=None):
        self.search_first, self.handoff = search_first, handoff
        self.extra_search_fields = extra_search_fields or {}
        self.seen = []

    def __call__(self, request):
        body = json.loads(request.content)
        schema = json.loads(body["messages"][0]["content"].split("schema:\n", 1)[1])
        data = json.loads(body["messages"][1]["content"])["untrusted_data"]
        self.seen.append((schema["title"], data))
        if schema["title"] == "ExtractionReview":
            result = {"accepted_indexes": [], "issues": []}  # Earlier/later phases sharing the call ledger.
        elif self.search_first and schema["title"] == "ResearchDecision" and len([s for s, _ in self.seen if s == "ResearchDecision"]) == 1:
            result = {"decision": {"action": "search_bank_materials", "query": "Прогноз ключевой ставки на конец года",
                       "metric": "policy_rate", "horizon": "2026-12-31", **self.extra_search_fields}}
        else:
            ids = self.handoff(data) if self.handoff else data["eligible_bank_claim_ids"]
            result = {"decision": {"action": "handoff", "bank_claim_ids": ids,
                                    "gaps": [] if ids else ["no_applicable_evidence"]}}
        return httpx.Response(200, json={"choices": [{"message": {"content": json.dumps(result)}, "finish_reason": "stop"}],
                                       "usage": {"prompt_tokens": 80, "completion_tokens": 50}})


def test_researcher_searches_observes_new_evidence_and_hands_off_real_ids(store):
    _, factory, _, settings = store
    bank = ScopedBank([copy.deepcopy(CURRENT), {**CURRENT, "id": "foreign-claim", "document_id": "forbidden-document"},
                       {**CURRENT, "id": "future-claim", "published_at": "2026-10-20"}])
    tools, gateway = make_tools(bank), ResearchGateway()
    with httpx.Client(transport=httpx.MockTransport(gateway)) as http:
        calls = Calls(Provider(factory, settings, http), "research-loop")
        result = BankResearchAgent(tools, calls, "Какой горизонт прогноза подтверждён?").run()
    assert [title for title, _ in gateway.seen] == ["ResearchDecision", "FinalResearchDecision"]
    before, after = [data for _, data in gateway.seen]
    assert before["eligible_bank_claim_ids"] == []
    assert after["eligible_bank_claim_ids"] and after["observations"]
    alias = after["eligible_bank_claim_ids"][0]
    assert tools.real_ids[alias] == CURRENT["id"]
    assert result["bank_claim_ids"] == [CURRENT["id"]] and result["status"] == "ready"
    assert tools.bank_claim_allowlist == {CURRENT["id"]}
    assert "foreign-claim" not in tools.registry and "future-claim" not in tools.registry
    assert calls.count == 2 and len(bank.requests) == 1
    _, scope, kwargs = bank.requests[0]
    assert scope.as_of == "2026-10-19"  # Previous completed UTC day, before any retrieval.
    assert scope.allowed_document_ids == frozenset({"allowed-document"})
    assert kwargs["metric"] == "policy_rate" and kwargs["horizon"] == "2026-12-31"
    assert tools.steps[-1]["action"] == "agent_handoff" and tools.steps[-1]["to_agent"] == "analyst"
    assert tools.steps[-2]["agent_role"] == "bank_researcher"
    with factory() as db:
        records = db.scalars(select(LLMCall)).all()
        assert len(records) == 2 and all(c.status == "completed" for c in records)


@pytest.mark.parametrize("bad_claim", [
    None,
    {**CURRENT, "id": "bad", "position_role": "previous_view"},
    {**CURRENT, "id": "bad", "verification_status": "ocr_unreviewed"},
    {**CURRENT, "id": "bad", "published_at": "2026-10-20"},
    {**CURRENT, "id": "bad", "forecast_horizon": "2025-12-31"},
    {**CURRENT, "id": "bad", "published_at": "2026-07-01"},
    {**CURRENT, "id": "bad", "published_at": "unknown"},
    {**CURRENT, "id": "bad", "forecast_horizon": "не указано"},
], ids=["foreign-id", "previous-view", "unreviewed", "future", "expired-horizon", "superseded", "invalid-publication-date", "invalid-horizon"])
def test_handoff_rejects_unknown_historical_unreviewed_and_superseded_claims(store, bad_claim):
    _, factory, _, settings = store
    tools = make_tools(ScopedBank())
    tools.registry[CURRENT["id"]] = {**CURRENT, "kind": "bank_claim"}
    if bad_claim:
        tools.registry["bad"] = {**bad_claim, "kind": "bank_claim"}
    gateway = ResearchGateway(search_first=False, handoff=lambda _: ["bad"])
    with httpx.Client(transport=httpx.MockTransport(gateway)) as http:
        researcher = BankResearchAgent(tools, Calls(Provider(factory, settings, http), "invalid-handoff"))
        with pytest.raises(AnalysisError, match="invalid_research_handoff"):
            researcher.run()
    assert tools.bank_claim_allowlist is None
    assert not any(s["action"] == "agent_handoff" for s in tools.steps)


def test_duplicate_real_claim_and_alias_cannot_bypass_handoff_validation(store):
    _, factory, _, settings = store
    tools = make_tools(ScopedBank())
    tools.registry[CURRENT["id"]] = {**CURRENT, "kind": "bank_claim"}
    gateway = ResearchGateway(search_first=False, handoff=lambda data: [CURRENT["id"], data["eligible_bank_claim_ids"][0]])
    with httpx.Client(transport=httpx.MockTransport(gateway)) as http:
        with pytest.raises(AnalysisError, match="invalid_research_handoff"):
            BankResearchAgent(tools, Calls(Provider(factory, settings, http), "duplicate-handoff")).run()


@pytest.mark.parametrize("earlier_calls, expected_research_calls", [(0, 2), (4, 2), (5, 1), (6, 0)])
def test_research_budget_is_shared_across_delegations_and_reserves_final_two_calls(store, earlier_calls, expected_research_calls):
    _, factory, _, settings = store
    bank, gateway = ScopedBank(), ResearchGateway()
    tools = make_tools(bank)
    with httpx.Client(transport=httpx.MockTransport(gateway)) as http:
        calls = Calls(Provider(factory, settings, http), "shared-research-budget")
        for index in range(earlier_calls):
            calls.ask(f"earlier_{index}", ExtractionReview, "Earlier pipeline stage", {})
        researcher = BankResearchAgent(tools, calls)
        first = researcher.run()
        assert researcher.decisions == expected_research_calls
        assert calls.count == earlier_calls + expected_research_calls <= settings.max_calls - 2
        # The analyst can delegate repeatedly; the same specialist never renews its allowance.
        for index in range(3):
            result = researcher.run(SearchBank(action="search_bank_materials", query=f"Уточнить прогноз, запрос {index}"))
            assert result["status"] == "limited"
            assert result["bank_claim_ids"] == first["bank_claim_ids"]
        assert calls.count == earlier_calls + expected_research_calls
        calls.ask("reserved_analyst", ExtractionReview, "Analyst stage", {})
        calls.ask("reserved_verifier", ExtractionReview, "Verifier stage", {})
        assert calls.count <= settings.max_calls
    assert len(bank.requests) == (1 if expected_research_calls == 2 else 0)


@pytest.mark.parametrize("delegate_request", [False, True])
def test_missing_index_never_spends_a_model_call(store, delegate_request):
    _, factory, _, settings = store
    tools = make_tools()
    def no_network(request):
        pytest.fail("Missing index must not call the gateway")
    with httpx.Client(transport=httpx.MockTransport(no_network)) as http:
        calls = Calls(Provider(factory, settings, http), "unavailable-index")
        request = SearchBank(action="search_bank_materials", query="Проверить позицию банка") if delegate_request else None
        result = BankResearchAgent(tools, calls).run(request)
    assert result["status"] == "unavailable" and result["bank_claim_ids"] == []
    assert calls.count == 0 and tools.bank_claim_allowlist == set()
    with factory() as db:
        assert not db.scalars(select(LLMCall)).all()


def test_research_action_cannot_override_access_or_as_of_before_tool_execution(store):
    _, factory, _, settings = store
    bank = ScopedBank()
    tools = make_tools(bank)
    gateway = ResearchGateway(extra_search_fields={"as_of": "2099-01-01", "allowed_document_ids": ["forbidden-document"]})
    with httpx.Client(transport=httpx.MockTransport(gateway)) as http:
        with pytest.raises(SchemaFailure):
            BankResearchAgent(tools, Calls(Provider(factory, settings, http), "scope-injection")).run()
    assert bank.requests == [] and tools.steps == []


def test_frozen_parent_bank_evidence_uses_parent_as_of_when_current_index_is_missing(store):
    _, factory, _, settings = store
    # The forecast has expired in the current year, but remains valid for this
    # immutable historical question. No new index data may replace its snapshot.
    tools = make_tools(as_of="2025-10-21T09:00:00+00:00")
    frozen = {**CURRENT, "id": "frozen-parent-claim", "published_at": "2025-09-11", "forecast_horizon": "2025-12-31", "kind": "bank_claim"}
    tools.registry[frozen["id"]] = frozen
    gateway = ResearchGateway()
    with httpx.Client(transport=httpx.MockTransport(gateway)) as http:
        calls = Calls(Provider(factory, settings, http), "frozen-parent-research")
        result = BankResearchAgent(tools, calls, "Какая позиция была доступна в исходном разборе?").run()
    assert result["bank_claim_ids"] == [frozen["id"]] and result["status"] == "ready"
    assert calls.count == 1 and gateway.seen[0][0] == "FinalResearchDecision"
    data = gateway.seen[0][1]
    assert data["as_of"] == tools.scope.as_of and data["bank_as_of"] == "2025-10-20"
    assert not any(step["action"] == "search_bank_materials" for step in tools.steps)
