"""0..5 relevance, capped by reviewed evidence on location, scope and activity.

Retrieval similarity and publisher reputation cannot lift an unknown dimension.
The runtime uses the same caps as the editorial annotation validator.
"""
from .inputs import industry_matches, instant, words

VERSION = "business-relevance-v2-0-to-5"
MIN_SELECTED_GRADE = 4
GRADE_DEFINITIONS = {
    "0": "Связь с деятельностью компании не установлена",
    "1": "Слабое тематическое совпадение или локальное событие вне подтверждённой географии",
    "2": "Смежная тема либо небольшой единичный кейс без существенного масштаба для компании",
    "3": "Профильная тема, но масштаб, применимость или география ещё требуют подтверждения",
    "4": "Существенный фактор для подтверждённой деятельности и применимого рынка компании",
    "5": "Прямая подтверждённая обязанность или критическое событие для компании с конкретным действием",
}


def assessment_cap(assessment):
    industry = assessment["industry_fit"]
    geography = assessment["geographic_fit"]
    materiality = assessment["materiality"]
    status = assessment["event_status"]
    if industry not in {"none","adjacent","direct"}:
        raise ValueError("invalid_industry_fit")
    if geography not in {"not_applicable","other_region","unknown","registration_only","national","global_market","confirmed_operations"}:
        raise ValueError("invalid_geographic_fit")
    if materiality not in {"none","small","unknown","material"}:
        raise ValueError("invalid_materiality")
    if status not in {"reported","proposal","forecast","in_force","announcement"}:
        raise ValueError("invalid_event_status")
    caps = [5]
    if industry == "none": caps.append(0)
    elif industry == "adjacent": caps.append(2)
    if geography == "other_region": caps.append(1)
    elif geography in {"unknown","registration_only","not_applicable"}: caps.append(3)
    if materiality == "none": caps.append(1)
    elif materiality == "small": caps.append(2)
    elif materiality == "unknown": caps.append(3)
    if status in {"proposal","announcement"}: caps.append(3)
    if assessment.get("timeliness_confirmed") is not True: caps.append(3)
    return min(caps)


def validate_assessment(grade, assessment):
    if type(grade) is not int or not 0 <= grade <= 5:
        raise ValueError("relevance_grade_must_be_0_to_5")
    if grade > assessment_cap(assessment):
        raise ValueError("grade_exceeds_geographic_or_materiality_cap")
    if grade >= 4 and not assessment.get("business_channel"):
        raise ValueError("selected_grade_needs_specific_business_channel")
    urgent_confirmed = ((assessment["event_status"] == "in_force" and assessment.get("applicability_confirmed") is True) or
                        (assessment["event_status"] == "reported" and assessment.get("direct_company_event_confirmed") is True))
    if grade == 5 and (not urgent_confirmed or not assessment.get("required_action")):
        raise ValueError("grade_5_requires_confirmed_applicability_and_action")
    return grade >= MIN_SELECTED_GRADE


def grade_event(company, event, scope, reasons):
    """No LLM calls: rate each coherent fact/channel, never mix unrelated facts.

    New extraction attributes have already passed an independent entailment
    check. Old extractions and raw mentions stay <=3 until they supply evidence.
    Registration alone never confirms operational presence, and raw revenue has
    no established unit: neither is used as a financial exposure estimate.
    """
    assessments = []
    for fact in event["facts"]:
        relevant = [r for r in reasons if fact["id"] in r["fact_ids"]]
        if not relevant:
            continue
        attrs = fact["attributes"]
        codes = {r["code"] for r in relevant}
        sector = (bool(company["bindings"]["industry"]) and any(
            industry_matches(company["industry"]["code"], prefix) for prefix in attrs["industry_codes"]))
        finance = bool(codes & {"credit_rate_channel", "deposit_rate_channel", "lease_rate_channel"})
        direct = sector or finance
        effect = attrs.get("business_effect", "unknown")
        market = attrs.get("market_scope", "unknown")
        geographic_scope = attrs.get("geographic_scope", "unknown")
        regions = {words(r) for r in attrs["regions"]}
        geography = "unknown"
        limits = ["company_revenue_unit_and_financial_exposure_unknown"]
        if geographic_scope == "national" and attrs.get("geography") == scope.market_country and not regions:
            geography = "national"
        elif geographic_scope == "global_market" and attrs.get("geography") == "GLOBAL" and not regions:
            geography = "global_market"
        elif geographic_scope in {"regional", "local"} or regions:
            if words(company["region"]) in regions:
                geography = "registration_only"
            limits.append("operational_geography_not_confirmed")
        else:
            limits.append("event_market_geography_not_confirmed")
        material = (direct and market == "sector_wide" and attrs.get("impact_scale") == "material" and (
            (sector and effect in {"core_market", "operating_requirement"}) or (finance and effect == "financing")))
        materiality = "small" if market == "single_case" or attrs.get("impact_scale") == "limited" else "material" if material else "unknown"
        status = attrs.get("event_status", "unknown")
        if status not in {"reported", "proposal", "forecast", "in_force", "announcement"}:
            status = "reported"
            limits.append("event_status_not_confirmed")
        effective = attrs.get("effective_from")
        if effective and instant(effective + "T23:59:59+00:00") > instant(scope.as_of):
            # Day precision cannot establish an obligation already in force at an earlier hour.
            status = "announcement"
            limits.append("requirement_effective_date_not_reached")
        source = next(s for s in event["sources"] if s["id"] == fact["source_id"])
        published = source["published_at"]  # Another source's date cannot establish this fact's freshness.
        age = (instant(scope.as_of) - instant(published)).total_seconds() / 86400 if published else None
        max_age = 90 if effect == "operating_requirement" else 45
        timely = age is not None and 0 <= age <= max_age
        horizon = attrs.get("forecast_horizon")
        if status == "forecast" and horizon and horizon < instant(scope.as_of).date().isoformat():
            timely = False
            limits.append("forecast_horizon_expired")
        if not timely:
            limits.append("publication_date_missing_or_revalidation_needed")
        if attrs.get("event_status", "unknown") == "unknown":
            materiality = "unknown"
        if attrs.get("kind") == "forecast" and status != "forecast":
            materiality = "unknown"
            limits.append("contradictory_forecast_status")
        if effect == "operating_requirement" and attrs.get("applicability") != "whole_industry":
            materiality = "unknown"
            limits.append("requirement_company_applicability_not_confirmed")
        if source["synthetic"]:
            timely = False
            limits.append("synthetic_source_not_for_live_feed")
        feature_ids = sorted({i for r in relevant for i in r["feature_ids"]})
        channel = ("Изменение условий финансирования: в профиле подтверждён соответствующий банковский продукт; условия договора требуют проверки."
                   if finance else "Событие затрагивает рынок основной деятельности, подтверждённой ОКВЭД компании."
                   if sector else "Подтверждён только общий рыночный либо смежный контекст; прямой канал не установлен.")
        assessment = {"industry_fit": "direct" if direct else "adjacent", "geographic_fit": geography,
            "materiality": materiality, "event_status": status, "timeliness_confirmed": timely,
            "business_channel": channel, "company_feature_ids": feature_ids,
            "fact_ids": [fact["id"]], "source_refs": [{"source_id": fact["source_id"],
                "source_version": fact["source_version"], "source_span": fact["source_span"]}],
            "scale_basis": market, "location_limitation": "Регистрация не подтверждает расположение операций или маршрутов.",
            "limitations": sorted(set(limits))}
        grade = min(4 if direct else 2, assessment_cap(assessment))
        if market == "single_case" and geographic_scope == "local" and geography == "unknown":
            grade = min(grade, 1)
        if (grade == 4 and sector and effect == "operating_requirement" and status == "in_force"
                and attrs.get("applicability") == "whole_industry"):
            grade = 5
            assessment.update(applicability_confirmed=True,
                required_action="Проверить исполнение действующего отраслевого требования, указанного в подтверждённом факте.")
        validate_assessment(grade, assessment)
        assessments.append((grade, assessment))
    if assessments:
        grade, assessment = max(assessments, key=lambda pair: (pair[0], pair[1]["fact_ids"][0]))
    else:
        raw_industry = any(r["code"] == "unverified_industry_mention" for r in reasons)
        grade = 2 if reasons else 0
        assessment = {"industry_fit": "direct" if raw_industry else "adjacent" if reasons else "none",
            "geographic_fit": "unknown", "materiality": "unknown", "event_status": "reported",
            "timeliness_confirmed": False, "business_channel": "Предварительное совпадение требует извлечения и проверки фактов.",
            "company_feature_ids": sorted({i for r in reasons for i in r["feature_ids"]}), "fact_ids": [],
            "source_refs": [ref for r in reasons for ref in r["source_refs"]],
            "limitations": ["reviewed_business_scope_missing", "operational_geography_not_confirmed"]}
    return {"relevance_grade": grade, "relevance_selected": grade >= MIN_SELECTED_GRADE,
        "relevance_assessment": assessment, "grading_version": VERSION}
