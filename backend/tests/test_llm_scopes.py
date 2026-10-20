"""Queue-to-worker access policy changes must precede any model context use."""
from dataclasses import replace
from unittest.mock import Mock

import pytest

from app.core import Company
from app.intelligence.storage import Insight
from app.intelligence.worker import process_next
from test_intelligence import queued_insight, runtime, store, stored_event


BANK_ID = "scope-bank-document"
BANK_TEXT = "Saved bank context must follow the worker's current access policy."


def parent_with_bank_context(store, monkeypatch):
    client, factory, headers, settings = store
    monkeypatch.setenv("BANK_ALLOWED_DOCUMENT_IDS", "*")
    event_id = stored_event(client, factory, headers["DEMO_A"])
    job_id = queued_insight(client, headers["DEMO_A"], event_id).json()["id"]
    engine, _ = runtime(factory, replace(settings, allowed_bank_ids=None))
    assert process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "succeeded", job
    card_id = job["result_id"]
    with factory.begin() as db:
        parent = db.get(Insight, card_id)
        parent.evidence = {**parent.evidence, "scope-bank-evidence": {
            "id": "scope-bank-evidence", "kind": "document_chunk", "document_id": BANK_ID,
            "text": BANK_TEXT}}
        # A question must still use the parent profile, rather than a fresh one.
        db.get(Company, "DEMO_A").profile_version = "changed-after-parent"
    return card_id, client.get("/api/v1/insights/" + card_id, headers=headers["DEMO_A"]).json()


def queue_question(client, headers, card_id):
    result = client.post(f"/api/v1/insights/{card_id}/questions",
                         headers={**headers["DEMO_A"], "Idempotency-Key": "scope-question-001"},
                         json={"question": "Какие условия договора нужно уточнить?"})
    assert result.status_code == 202, result.text
    return result.json()["id"]


@pytest.mark.parametrize("allowed", [frozenset(), frozenset({"other-bank-document"})])
def test_queued_question_rechecks_bank_access_before_provider(store, monkeypatch, allowed):
    client, factory, headers, settings = store
    card_id, _ = parent_with_bank_context(store, monkeypatch)
    job_id = queue_question(client, headers, card_id)
    # The API authorized the enqueue, then a newer worker policy removes access.
    monkeypatch.setenv("BANK_ALLOWED_DOCUMENT_IDS", ",".join(sorted(allowed)))
    assert client.get("/api/v1/insights/" + card_id, headers=headers["DEMO_A"]).status_code == 403
    engine, gateway = runtime(factory, replace(settings, allowed_bank_ids=allowed))
    engine.provider.generate = Mock(wraps=engine.provider.generate)
    assert process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "failed" and job["error"] == "evidence_outside_scope", job
    engine.provider.generate.assert_not_called()
    assert gateway.seen == []


@pytest.mark.parametrize("allowed", [None, frozenset({BANK_ID})])
def test_question_preserves_authorized_bank_context_and_frozen_profile(store, monkeypatch, allowed):
    client, factory, headers, settings = store
    card_id, parent = parent_with_bank_context(store, monkeypatch)
    monkeypatch.setenv("BANK_ALLOWED_DOCUMENT_IDS", "*" if allowed is None else BANK_ID)
    job_id = queue_question(client, headers, card_id)
    engine, gateway = runtime(factory, replace(settings, allowed_bank_ids=allowed))
    assert process_next(factory, engine)
    job = client.get("/api/v1/jobs/" + job_id, headers=headers["DEMO_A"]).json()
    assert job["status"] == "succeeded", job
    answer = client.get("/api/v1/answers/" + job["result_id"], headers=headers["DEMO_A"])
    assert answer.status_code == 200, answer.text
    assert answer.json()["profile_version"] == parent["profile_version"]
    assert answer.json()["as_of"] == parent["as_of"]
    assert answer.json()["role"] == parent["role"]
    decision = next(data for title, data in gateway.seen if title == "Decision")
    assert any(item["text"] == BANK_TEXT for item in decision["evidence"])
    evidence = client.get("/api/v1/evidence/scope-bank-evidence",
                          params={"insight_id": card_id, "answer_id": job["result_id"]}, headers=headers["DEMO_A"])
    assert evidence.status_code == 200, evidence.text
    assert evidence.json()["document_id"] == BANK_ID
