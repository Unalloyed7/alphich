"""Machine QA and immutable diagnostic queues, without any API/model download."""
import copy
import tempfile
from pathlib import Path

import pytest

from alphich_kb.common import KnowledgeError, digest, file_hash
from alphich_kb.relevance_ml.automatic import (
    QUEUE_SCHEMA, attempted_pairs, build_matched_train, machine_consensus, queued_rows, select_candidates,
    score_candidates, validate_labeling_input,
)
from alphich_kb.relevance_ml.api_runner import run_api_batch
from alphich_kb.relevance_ml.dataset import input_hash, validate_pair, write_dataset, write_private_json
from alphich_kb.relevance_ml.teacher import (
    exact_quote_units, judge_pair, materialize_quote_ids, prepare_judgement, validate_checkpoint, validate_judgement,
)
from test_relevance_api_runner import MODEL, LedgerProvider, pair

OTHER = "qwen/qwen3.6-35b-a3b"


def proposals(index=1):
    row = pair(index)
    a = judge_pair(row, LedgerProvider(), batch_id="primary", model_id=MODEL)
    b = judge_pair(row, LedgerProvider(), batch_id="blind-review", model_id=OTHER)
    return a, b


def test_blind_prompts_cannot_see_first_model_label_or_reason():
    a, _ = proposals()
    request = prepare_judgement(a, prompt_variant="exact-evidence")
    assert "label" not in request["data"]
    assert request["data"] == prepare_judgement(pair())["data"]
    assert request["prompt_hash"] != prepare_judgement(pair())["prompt_hash"]
    assert prepare_judgement(pair())["prompt_version"] == "relevance-judge-v2-business-fit"


def test_quote_ids_copy_original_text_and_source_ids_without_silent_truncation():
    row = pair()
    original = row["article"]["text"] + "\n  Хвост\t" * 250 + "!"
    units = exact_quote_units(original)
    assert "".join(unit["text"] for unit in units) == original
    assert all(original[u["start"]:u["end"]] == u["text"] for u in units)
    plan = prepare_judgement(row, prompt_variant="quote-ids")
    assert "text" not in plan["data"]["article"]
    assert plan["data"]["company"]["facts"][0]["id"] == "c1"
    from test_relevance_api_runner import judgement
    wire = judgement()
    wire["news_quote_ids"] = ["q1"]; wire.pop("news_quotes")
    wire["temporal_quote_ids"] = []; wire.pop("temporal_quotes")
    wire["company_fact_ids"] = ["c1"]
    result = materialize_quote_ids(wire, validate_pair(row))
    assert validate_judgement(result, row)["news_quotes"] == [row["article"]["text"]]
    assert result["company_fact_ids"] == ["okved"]
    wire["news_quote_ids"] = ["q9999"]
    with pytest.raises(KnowledgeError, match="invalid evidence"):
        materialize_quote_ids(wire, row)


def test_business_only_labels_never_claim_current_temporal_eligibility():
    from test_relevance_api_runner import judgement
    row = pair()
    wire = judgement()
    wire["news_quote_ids"] = ["q1"]
    wire["company_fact_ids"] = ["c1"]
    for key in ("news_quotes", "temporal_quotes", "timeliness", "temporal_reason", "valid_from", "valid_until"):
        wire.pop(key)
    plan = prepare_judgement(row, prompt_variant="business-only")
    assert "timeliness" not in plan["response_schema"]["properties"]
    value = validate_judgement(materialize_quote_ids(wire, row, business_only=True), row)
    assert value["grade"] == 4 and value["timeliness"] == "unknown"
    assert not value["temporal_quotes"] and value["valid_until"] is None


def test_case_calibration_changes_prompt_identity_without_changing_source_or_old_contract():
    from alphich_kb.personalization.criteria import CASE_RELEVANCE_GUIDE
    from app.intelligence.relevance import SYSTEM
    old = prepare_judgement(pair(), prompt_variant="business-only")
    new = prepare_judgement(pair(), prompt_variant="case-calibrated")
    assert old["input_hash"] == new["input_hash"] and old["data"] == new["data"]
    assert old["prompt_hash"] != new["prompt_hash"]
    assert CASE_RELEVANCE_GUIDE in new["system"] and CASE_RELEVANCE_GUIDE in SYSTEM
    assert "условия договора" in CASE_RELEVANCE_GUIDE


def test_new_case_criteria_cannot_silently_reuse_older_negative_teacher_labels():
    from alphich_kb.relevance_ml.model import train_model
    from test_relevance_model import MANIFEST, pair as model_pair
    import numpy as np
    rows = [model_pair("old", 0, origin="teacher"), model_pair("new", 4, origin="teacher")]
    rows[1]["label"]["criteria_version"] = "case-business-significance-v1"
    with pytest.raises(KnowledgeError, match="cannot be silently mixed"):
        train_model(rows, np.asarray([[0, 1], [1, 0]]), feature_manifest=MANIFEST, allow_teacher=True)
    a, b = proposals()
    a["label"]["criteria_version"] = "case-business-significance-v1"
    accepted, abstained, _ = machine_consensus([a], [b], policy="binary")
    assert not accepted and "grading_criteria_disagreement" in abstained[0]["reasons"]


def test_exact_evidence_checkpoints_resume_with_their_own_contract():
    with tempfile.TemporaryDirectory() as folder:
        provider = LedgerProvider()
        kwargs = dict(destination=folder, batch_id="new-contract", model_id=MODEL,
                      ledger_reader=provider.ledger, prompt_variant="exact-evidence")
        run_api_batch([pair()], provider, **kwargs)
        run_api_batch([pair()], provider, **kwargs)
        assert len(provider.dispatched) == 1
        assert len(attempted_pairs([folder])) == 1
        with pytest.raises(KnowledgeError, match="different immutable inputs"):
            run_api_batch([pair()], provider, **{**kwargs, "prompt_variant": "legacy"})


def test_consensus_accepts_only_exact_machine_agreement_and_keeps_teacher_origin():
    a, b = proposals()
    accepted, abstained, report = machine_consensus([a], [b])
    assert len(accepted) == 1 and not abstained
    label = accepted[0]["label"]
    assert label["origin"] == "teacher" and label["independently_reviewed"] is False
    assert label["machine_review"]["reviewer"]["labeler"] == OTHER
    assert report["accuracy"] is None and report["human_benchmark_eligible"] is False


def test_binary_agreement_does_not_pretend_ordinal_or_temporal_consensus():
    a, b = proposals()
    a["label"]["grade"] = a["label"]["judgement"]["grade"] = 1
    b["label"]["grade"] = b["label"]["judgement"]["grade"] = 3
    b["label"]["judgement"]["geography"] = "unknown"
    accepted, abstained, report = machine_consensus([a], [b], policy="binary")
    assert len(accepted) == 1 and not abstained
    assert accepted[0]["label"]["grade"] == 1
    review = accepted[0]["label"]["machine_review"]
    assert review["agreement_target"] == "grade>=4" and review["grade_interval"] == [1, 3]
    assert report["human_benchmark_eligible"] is False
    a["label"]["grade"] = a["label"]["judgement"]["grade"] = 4
    assert not machine_consensus([a], [b], policy="binary")[0]


@pytest.mark.parametrize("change,reason", [
    ("grade", "grade_disagreement"), ("scope", "scope_or_temporal_disagreement"),
    ("model", "same_model_not_cross_model_review"), ("missing", "missing_blind_review"),
])
def test_machine_disagreement_abstains_instead_of_averaging_or_creating_a_negative(change, reason):
    a, b = proposals()
    if change == "grade":
        b["label"]["grade"] = b["label"]["judgement"]["grade"] = 3
    if change == "scope":
        b["label"]["judgement"]["event_status"] = "announced"
    if change == "model":
        b["label"]["labeler"] = MODEL
    accepted, abstained, _ = machine_consensus([a], [] if change == "missing" else [b])
    assert not accepted and reason in abstained[0]["reasons"]


def test_fabricated_quotes_and_rebound_company_snapshots_cannot_be_silver():
    a, b = proposals()
    b["label"]["judgement"]["news_quotes"] = ["Отсутствует в источнике"]
    with pytest.raises(KnowledgeError, match="absent"):
        machine_consensus([a], [b])
    a, b = proposals()
    b["company"]["profile_version"] = "changed"
    with pytest.raises(KnowledgeError):
        machine_consensus([a], [b])


def test_selection_balances_stories_roles_and_excludes_paid_failures_without_labels():
    rows = []
    for i in range(20):
        for role in ("cfo", "ceo"):
            row = pair(i + 1)
            row.pop("pair_id")
            row["role"] = role
            row["company"]["company_id"] = "company-" + str(i % 6)
            row["article"]["text"] += " Обсуждают законопроект, пилотный проект."
            row["article"]["cluster_id"] = "story-" + str(i)
            rows.append(validate_pair(row))
    scores = [i / len(rows) for i in range(len(rows))]
    excluded = {rows[-1]["pair_id"]}
    queue, report = select_candidates(rows, scores, limit=16, excluded=excluded)
    assert len(queue) == 16 and report["unique_clusters"] == 16
    assert abs(report["role_counts"]["ceo"] - report["role_counts"]["cfo"]) <= 1
    assert not excluded & {r["pair_id"] for r in queue}
    assert {r["arm"] for r in queue} >= {"random_control", "hard_negative_candidate", "boundary", "high_similarity"}
    assert all("label" not in r for r in queue)
    assert queue == select_candidates(rows, scores, limit=16, excluded=excluded)[0]
    assert len(select_candidates(rows, scores, limit=4)[0]) <= 4
    with pytest.raises(KnowledgeError):
        select_candidates(rows, [float("nan")] * len(rows))


def test_sparse_queue_checks_file_and_each_original_offset_before_dispatch():
    with tempfile.TemporaryDirectory() as folder:
        source, path = Path(folder) / "train.jsonl", Path(folder) / "queue.json"
        rows = [pair(i) for i in range(8)]
        write_dataset(source, rows)
        requests, _ = select_candidates(rows, [0.1 * i for i in range(8)], limit=4)
        queue = {"schema_version": QUEUE_SCHEMA, "source_sha256": file_hash(source), "requests": requests}
        queue["queue_id"] = "queue_" + digest(queue)
        write_private_json(path, queue)
        assert [r["pair_id"] for r in queued_rows(source, path)] == [r["pair_id"] for r in requests]
        source.write_text(source.read_text() + "\n")
        with pytest.raises(KnowledgeError, match="checksum"):
            queued_rows(source, path)


def test_settled_validation_failure_continues_and_final_status_does_not_keep_stale_error():
    with tempfile.TemporaryDirectory() as folder:
        provider = LedgerProvider(["invalid", "ok"])
        kwargs = dict(destination=folder, batch_id="settled-errors", model_id=MODEL,
                      ledger_reader=provider.ledger, continue_settled_validation_errors=True)
        result = run_api_batch([pair(1), pair(2)], provider, **kwargs)
        assert result["status"] == "completed_with_errors" and result["stop_reason"] is None
        assert result["pending"] == 0 and result["unlabeled_failures"] == 1 and result["completed"] == 1
        assert result["known_cost_rub"] == "3"
        assert len(attempted_pairs([folder])) == 2
        run_api_batch([pair(1), pair(2)], provider, **kwargs)
        assert len(provider.dispatched) == 2


@pytest.mark.parametrize("outcome", ["llm_timeout_outcome_unknown", "llm_http_503", "budget_exceeded"])
def test_continue_flag_never_retries_or_ignores_http_unknown_cost_or_budget(outcome):
    with tempfile.TemporaryDirectory() as folder:
        provider = LedgerProvider([outcome, "ok"])
        result = run_api_batch([pair(1), pair(2)], provider, destination=folder, batch_id="must-stop",
                              model_id=MODEL, ledger_reader=provider.ledger,
                              continue_settled_validation_errors=True)
        assert result["status"] == "stopped" and len(provider.dispatched) == 1


def test_validation_failure_without_accounting_record_cannot_continue():
    with tempfile.TemporaryDirectory() as folder:
        provider = LedgerProvider(["invalid", "ok"])
        result = run_api_batch([pair(1), pair(2)], provider, destination=folder, batch_id="missing-ledger",
                              model_id=MODEL, ledger_reader=lambda run_id: [],
                              continue_settled_validation_errors=True)
        assert result["status"] == "stopped" and len(provider.dispatched) == 1
        run_api_batch([pair(1), pair(2)], provider, destination=folder, batch_id="missing-ledger",
                      model_id=MODEL, ledger_reader=lambda run_id: [], continue_settled_validation_errors=True)
        assert len(provider.dispatched) == 1


def test_missing_failed_receipt_makes_cost_per_accepted_label_unknown():
    with tempfile.TemporaryDirectory() as folder:
        provider = LedgerProvider(["ok", "invalid"])
        first_run = []

        def ledger(run_id):
            if not first_run:
                first_run.append(run_id)
            return provider.ledger(run_id) if run_id == first_run[0] else []

        result = run_api_batch([pair(1), pair(2)], provider, destination=folder, batch_id="missing-failed-receipt",
            model_id=MODEL, ledger_reader=ledger, continue_settled_validation_errors=True)
        assert result["completed"] == 1 and result["unaccounted_attempts"] == 1
        assert result["cost_per_accepted_label_rub"] is None


class TinyEncoder:
    dimension, max_tokens, space_id = 4, 128, "synthetic-test-space"

    def count_tokens(self, text):
        return len(text) + 2

    def encode_query(self, text):
        import numpy as np
        assert self.count_tokens("query: " + text) <= self.max_tokens
        return np.asarray([1, 2, 3, 4], dtype=np.float32)

    def encode_passages(self, texts, batch_size=4):
        import numpy as np
        assert all(self.count_tokens(text) <= self.max_tokens for text in texts)
        return np.asarray([[4, 3, 2, 1] for _ in texts], dtype=np.float32)


def test_candidate_scoring_covers_full_article_and_ignores_labels_and_registration():
    a, _ = proposals()
    a["article"]["text"] += " Дополнение." * 100
    a.pop("pair_id")
    a["label"]["input_hash"] = input_hash(a)
    a = validate_pair(a)
    score, details, _ = score_candidates([a], TinyEncoder())
    assert details[0]["full_article_chunks"] > 1 and details[0]["article_coverage"] == "complete"
    b = copy.deepcopy(a)
    b["label"]["grade"] = 0
    b["company"]["facts"].append({"id": "location", "name": "registration_region", "value": "Париж", "source_ref": "fixture"})
    b.pop("pair_id"); b["label"]["input_hash"] = input_hash(b)
    assert score_candidates([b], TinyEncoder())[0] == score


def test_rematched_train_preserves_the_original_pilot_and_rejects_tampering():
    from test_relevance_api_pilot import articles, company
    from alphich_kb.relevance_ml.pilot import prepare_pilot, load_pilot
    from alphich_kb.common import read_json
    with tempfile.TemporaryDirectory() as folder:
        pilot, supplement = Path(folder) / "pilot", Path(folder) / "supplement"
        prepare_pilot(articles(), [company("a"), company("b", code="49.41"), company("c", code="47.11")],
                      pilot, as_of="2026-10-24T00:00:00Z", limit=60)
        original_hash = file_hash(pilot / "manifest.json")
        _, splits = load_pilot(pilot / "manifest.json")
        excluded = {splits["train"][0]["pair_id"]}
        build_matched_train(pilot / "manifest.json", supplement, TinyEncoder(), excluded=excluded)
        manifest = validate_labeling_input(supplement / "manifest.json", supplement / "train.jsonl")
        assert not excluded & set(manifest["input_hashes"])
        assert file_hash(pilot / "manifest.json") == original_hash
        assert manifest["parent_plan_sha256"] == original_hash
        with pytest.raises(KnowledgeError):
            validate_labeling_input(supplement / "manifest.json", pilot / "test.jsonl")
        (supplement / "train.jsonl").write_text((supplement / "train.jsonl").read_text() + "\n")
        with pytest.raises(KnowledgeError, match="changed"):
            validate_labeling_input(supplement / "manifest.json", supplement / "train.jsonl")


def test_new_source_articles_cannot_smuggle_a_holdout_story_into_train():
    from test_relevance_api_pilot import articles, company
    from alphich_kb.relevance_ml.pilot import prepare_pilot, load_pilot
    with tempfile.TemporaryDirectory() as folder:
        pilot, supplement = Path(folder) / "pilot", Path(folder) / "supplement"
        prepare_pilot(articles(), [company("a"), company("b", code="49.41"), company("c", code="47.11")],
                      pilot, as_of="2026-10-24T00:00:00Z", limit=60)
        _, splits = load_pilot(pilot / "manifest.json")
        forbidden = splits["test"][0]["article"]
        fresh = copy.deepcopy(splits["train"][0]["article"])
        fresh.update(article_id="fresh", cluster_id="fresh", url="https://example.test/fresh", text="Полностью новая статья.")
        build_matched_train(pilot / "manifest.json", supplement, TinyEncoder(), articles=[forbidden, fresh])
        manifest = validate_labeling_input(supplement / "manifest.json", supplement / "train.jsonl")
        assert len(manifest["article_exclusions"]) == 1
        assert len(manifest["input_hashes"]) == 6
