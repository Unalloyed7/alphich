"""No-LLM diagnostics: citation structure is not semantic truth or business usefulness."""
VERSION = "content-quality-v1"


def content_quality(card, registry):
    claims = card.get("claims", [])
    kinds = {"news_fact": "event_fact", "company_fact": "company_feature", "bank_view": "bank_claim"}
    factual = [c for c in claims if c.get("kind") in kinds]
    inferences = [c for c in claims if c.get("kind") == "model_inference"]
    actions = card.get("next_steps", [])
    company_id = card["company_id"]

    def valid_refs(claim):
        refs = claim.get("evidence_ids", [])
        return bool(refs) and all(i in registry and (registry[i].get("kind") not in {"event_fact", "company_feature"}
                    or registry[i].get("company_id") == company_id) for i in refs)

    exact = sum(valid_refs(c) and any(registry[i].get("kind") == kinds[c["kind"]]
                and registry[i].get("text") == c.get("text") for i in c["evidence_ids"]) for c in factual)
    personal = sum(valid_refs(c) and {"event_fact", "company_feature"} <=
                   {registry[i].get("kind") for i in c["evidence_ids"]} and bool(c.get("conditions")) for c in inferences)
    inference_ids = {c["id"] for c in inferences}
    traced = sum(bool(a.get("conditions")) and bool(a.get("basis_claim_ids")) and
                 set(a["basis_claim_ids"]) <= inference_ids for a in actions)

    def metric(passed, total):
        return {"passed": int(passed), "total": total, "ratio": passed / total if total else None}

    verification = card.get("verification", {})
    return {"version": VERSION, "company_id": company_id,
        "factual_quote_match": metric(exact, len(factual)),
        "inference_evidence_structure": metric(personal, len(inferences)),
        "action_traceability": metric(traced, len(actions)),
        "selection_grade": card.get("relevance_grade"), "missing_data_count": len(card.get("missing_data", [])),
        "automated_review": {"status": verification.get("status"), "issues": verification.get("issues", [])},
        "human_review": {"status": "not_recorded"}, "semantic_correctness": None, "business_usefulness": None,
        "limitations": ["structure_does_not_establish_semantic_support", "automated_review_is_not_independent_human_review",
                        "ratios_with_zero_denominator_are_not_applicable", "no_single_quality_score"]}
