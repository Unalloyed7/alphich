"""Explicit 0..5 editorial rubric; assessments require evidence, not embedding scores.

This validates annotations and proposed ratings. It does not infer geographic
presence, turnover units, company identity or financial exposure from keywords.
"""
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


def validate_assessment(grade, assessment):
    if type(grade) is not int or not 0 <= grade <= 5:
        raise ValueError("relevance_grade_must_be_0_to_5")
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
    if grade > min(caps):
        raise ValueError("grade_exceeds_geographic_or_materiality_cap")
    if grade >= 4 and not assessment.get("business_channel"):
        raise ValueError("selected_grade_needs_specific_business_channel")
    urgent_confirmed = ((status == "in_force" and assessment.get("applicability_confirmed") is True) or
                        (status == "reported" and assessment.get("direct_company_event_confirmed") is True))
    if grade == 5 and (not urgent_confirmed or not assessment.get("required_action")):
        raise ValueError("grade_5_requires_confirmed_applicability_and_action")
    return grade >= MIN_SELECTED_GRADE
