"""Model-facing projections; immutable evidence and audit records stay complete."""
from __future__ import annotations


def business_focus_context(focus):
    """Pass evidence selection and open policy once, not repeated instructions."""
    return {key: focus[key] for key in (
        "mode", "channel", "driver", "event_title", "news_ids", "company_ids",
        "max_inferences", "unknowns", "selection_policy") if key in focus}


def relevance_context(selection):
    """A prior judgment is context, not another copy of E5 diagnostics or proof."""
    assessment = selection.get("relevance_assessment") or {}
    judgment = assessment.get("judgment") or assessment
    result = {"grade": selection.get("relevance_grade"),
              "selected": selection.get("relevance_selected"),
              "prior_assessment_is_not_evidence": True}
    for key in ("reason", "industry_fit", "geographic_fit", "materiality", "event_status",
                "missing_data", "required_action", "limitations", "news_fact_ids", "company_feature_ids"):
        if key in judgment:
            result[key] = judgment[key]
        elif key in assessment:
            result[key] = assessment[key]
    return result


def observation_context(observation):
    """Avoid repeating full registry metadata after a tool reads existing sources.

    Source text remains an explicitly marked excerpt. Full originals remain in
    the scoped registry and audit steps for inspection; IDs are unchanged.
    """
    if not isinstance(observation, dict) or "evidence" not in observation:
        return observation
    records = []
    for item in observation["evidence"]:
        summary = {key: item[key] for key in (
            "id", "kind", "text", "name", "value", "unit", "document_id", "company_id", "event_id",
            "source_id", "source_version", "source_span", "published_at", "source_available_at", "metric",
            "forecast_horizon", "position_role", "verification_status", "ocr_checked", "attributed_to",
            "quality_flags") if key in item}
        text = summary.get("text", "")
        if isinstance(text, str) and len(text) > 1600:
            summary["text"] = text[:1600]
            summary["prompt_excerpt_only"] = True
        summary.update(metadata_omitted=True, original_evidence_retained=True)
        records.append(summary)
    return {**observation, "evidence": records, "model_context_summary": True,
            "original_evidence_retained": True}
