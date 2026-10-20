"""Deterministic provenance checks, followed by a bounded semantic review."""
import json
import re
from decimal import Decimal

LABELS = {"okved_code": "ОКВЭД", "okved_name": "Основная деятельность", "region_registration": "Регион регистрации",
          "revenue_raw": "Выручка в исходных единицах", "fts_countries": "Страны в сведениях ФТС",
          "CL_PRODUCT_LOA_MARK": "Кредитный продукт", "CL_PRODUCT_DEP_MARK": "Депозитный продукт",
          "CL_PRODUCT_LEA_MARK": "Лизинговый продукт", "CL_PRODUCT_GAR_MARK": "Гарантийный продукт",
          "CL_PRODUCT_RKO_MARK": "Расчётное обслуживание", "CL_PRODUCT_ZPP_MARK": "Зарплатный проект",
          "FACTORING_MARK": "Факторинг"}


def company_evidence(profile):
    result = {}
    for feature in profile.get("features", []):
        if feature["name"] == "person_links":
            continue
        value = feature["value"]
        display = "да" if value is True else "нет" if value is False else "неизвестно" if value is None else (
            ", ".join(value) if isinstance(value, list) and all(isinstance(v, str) for v in value)
            else json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
        result[feature["id"]] = {**feature, "kind": "company_feature", "company_id": profile.get("company_id", profile.get("id")),
                                 "text": f"{LABELS.get(feature['name'], feature['name'])}: {display}.",
                                 "quality_flags": ["company_snapshot_date_unknown"]}
    return result


def numbers(text):
    return {Decimal(n.replace(",", ".")).normalize() for n in re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", text)}


def verify_draft(draft, registry):
    issues = []
    claims = {claim.id: claim for claim in draft.claims}
    if len(claims) != len(draft.claims):
        issues.append("duplicate_claim_ids")
    kinds = {"news_fact": "event_fact", "company_fact": "company_feature", "bank_view": "bank_claim"}
    for claim in draft.claims:
        evidence = [registry[i] for i in claim.evidence_ids if i in registry]
        if len(evidence) != len(claim.evidence_ids):
            issues.append(f"{claim.id}:unknown_evidence")
            continue
        if claim.kind in kinds:
            matching = [e for e in evidence if e["kind"] == kinds[claim.kind] and e["text"] == claim.text]
            if not matching:
                issues.append(f"{claim.id}:facts_must_use_exact_server_text")
            if claim.kind == "bank_view" and any(e.get("position_role") != "current_in_document"
                                                for e in matching if e.get("kind") == "bank_claim"):
                issues.append(f"{claim.id}:bank_claim_is_not_current_in_document")
            if claim.kind == "bank_view":
                dimensions = ("metric", "forecast_horizon", "geography", "scenario", "attributed_to", "position_role")
                for bank_claim in matching:
                    newer = [e for e in registry.values() if e["kind"] == "bank_claim" and
                             all(e.get(k) == bank_claim.get(k) for k in dimensions) and
                             e.get("published_at", "") > bank_claim.get("published_at", "")]
                    if newer:
                        issues.append(f"{claim.id}:bank_claim_superseded_in_context")
        else:
            evidence_kinds = {e["kind"] for e in evidence}
            if not {"event_fact", "company_feature"} <= evidence_kinds or not claim.conditions:
                issues.append(f"{claim.id}:personal_inference_needs_company_news_and_conditions")
            # OCR without span review cannot establish an exact numeric input.
            trusted_text = " ".join(e["text"] for e in evidence if e["kind"] != "document_chunk" or e.get("ocr_checked"))
            if not numbers(claim.text) <= numbers(trusted_text):
                issues.append(f"{claim.id}:unsupported_number")
            if numbers(claim.text):
                issues.append(f"{claim.id}:numeric_inference_must_be_quoted_as_fact")
            if re.search(r"альфа.?банк|по мнению банка|банк гарантирует", claim.text, re.I):
                issues.append(f"{claim.id}:bank_attribution_requires_bank_view")
        if len(set(claim.evidence_ids)) != len(claim.evidence_ids):
            issues.append(f"{claim.id}:duplicate_evidence")
    for ref in draft.risks + draft.opportunities:
        if ref not in claims or claims[ref].kind != "model_inference":
            issues.append("risk_or_opportunity_must_reference_inference")
    for index, action in enumerate(draft.next_steps):
        if not set(action.basis_claim_ids) <= claims.keys():
            issues.append(f"action_{index}:unknown_claim")
        basis = " ".join(claims[i].text for i in action.basis_claim_ids if i in claims)
        if not numbers(action.text) <= numbers(basis):
            issues.append(f"action_{index}:unsupported_number")
    free_text = draft.missing_data + [c.text for c in draft.claims if c.kind == "model_inference"]
    free_text += [condition for c in draft.claims for condition in c.conditions]
    free_text += [a.text for a in draft.next_steps] + [condition for a in draft.next_steps for condition in a.conditions]
    if any(re.search(r"https?://|www\.|javascript:|data:", value, re.I) for value in free_text):
        issues.append("model_links_must_use_server_evidence")
    return sorted(set(issues))
