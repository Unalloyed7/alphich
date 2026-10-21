"""Deterministic provenance checks, followed by a bounded semantic review."""
import json
import re
from decimal import Decimal

from .schemas import CardDraft

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
        if feature["name"] in {"payment_incoming", "payment_outgoing"}:
            label = "Входящие" if feature["name"] == "payment_incoming" else "Исходящие"
            sectors = "; ".join(f"{s.get('code')}: {str(s.get('name') or '')[:70]}, доля {s.get('share_of_all_flow')}"
                                for s in value.get("known_sectors", [])[:3])
            text = (f"{label} платежи: сумма в исходных единицах {value.get('amount_raw')}; "
                    f"количество платежей {value.get('payment_count')}; контрагентов {value.get('counterparty_count')}; "
                    f"доля крупнейшего {value.get('top_counterparty_share')}; "
                    f"покрытие известными профилями {value.get('known_profile_amount_share')}. "
                    "Доли от 0 до 1. Валюта, период и назначение платежей неизвестны. "
                    f"До трёх отраслей известных контрагентов: {sectors or 'нет данных'}.")
        else:
            text = None
        display = "да" if value is True else "нет" if value is False else "неизвестно" if value is None else (
            ", ".join(value) if isinstance(value, list) and all(isinstance(v, str) for v in value)
            else json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else str(value))
        result[feature["id"]] = {**feature, "kind": "company_feature", "company_id": profile.get("company_id", profile.get("id")),
                                 "text": text or f"{LABELS.get(feature['name'], feature['name'])}: {display}.",
                                 "quality_flags": ["company_snapshot_date_unknown"]}
    return result


def numbers(text):
    return {Decimal(n.replace(",", ".")).normalize() for n in re.findall(r"(?<!\w)\d+(?:[.,]\d+)?", text)}


def materialize_draft(draft, registry):
    """The model selects facts by ID; only the server supplies factual text/types."""
    issues = []
    kinds = {"event_fact": "news_fact", "company_feature": "company_fact", "bank_claim": "bank_view"}
    inference_ids = [i.id for i in draft.inferences]
    if len(inference_ids) != len(set(inference_ids)):
        issues.append("duplicate_inference_ids")
    if len(draft.fact_evidence_ids) != len(set(draft.fact_evidence_ids)):
        issues.append("duplicate_fact_evidence_ids")
    factual_ids = list(draft.fact_evidence_ids)
    for inference in draft.inferences:
        for evidence_id in inference.evidence_ids:
            item = registry.get(evidence_id)
            if item and item["kind"] in kinds and evidence_id not in factual_ids:
                factual_ids.append(evidence_id)
    for evidence_id in factual_ids:
        item = registry.get(evidence_id)
        if not item or item["kind"] not in kinds:
            issues.append("fact_reference_requires_event_company_or_reviewed_bank_claim")
    for ref in draft.risks + draft.opportunities + [r for a in draft.next_steps for r in a.basis_inference_ids]:
        if ref not in inference_ids:
            issues.append("unknown_inference_id:" + ref)
    if not factual_ids and not draft.inferences:
        issues.append("card_requires_at_least_one_claim")
    if len(factual_ids) + len(draft.inferences) > 12:
        issues.append("too_many_card_claims")
    if issues:
        return None, sorted(set(issues))
    claims = [{"id": f"c{index+1}", "kind": kinds[registry[ref]["kind"]], "text": registry[ref]["text"],
               "evidence_ids": [ref], "conditions": []} for index, ref in enumerate(factual_ids)]
    mapped = {item.id: f"c{len(claims)+index+1}" for index, item in enumerate(draft.inferences)}
    claims += [{"id": mapped[item.id], "kind": "model_inference", "text": item.text,
                "evidence_ids": item.evidence_ids, "conditions": item.conditions} for item in draft.inferences]
    value = {"claims": claims, "risks": [mapped[i] for i in draft.risks],
             "opportunities": [mapped[i] for i in draft.opportunities],
             "next_steps": [{"text": a.text, "conditions": a.conditions,
                             "basis_claim_ids": [mapped[i] for i in a.basis_inference_ids]} for a in draft.next_steps],
             "missing_data": draft.missing_data}
    try:
        return CardDraft.model_validate(value), []
    except ValueError:
        return None, ["resolved_card_schema_invalid"]


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
