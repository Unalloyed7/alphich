"""Shared, conservative interpretation of organizer company source fields.

Labels remain labels. In particular an age band does not establish a company's
growth stage, and NEGATIVEMARK has no documented credit or bankruptcy meaning.
This module performs no database or network access.
"""
from __future__ import annotations

import re
from collections.abc import Mapping

FACTOR_VERSION = "organizer-company-factors-v1"
ORGANIZER_FACTOR_COLUMNS = {
    "organization_type": "ORG_TYPE",
    "ownership_form": "OKFS_NAME",
    "company_age_band": "COMPANY_AGE_YEARS_GRADATION",
    "business_stage_label": "ECO_MARK_LABEL",
    "negative_mark_raw": "NEGATIVEMARK",
    "employees_fns_band": "AMOUNT_WORKERS_FNS_GRADATION",
    "employees_spark_band": "AMOUNT_WORKERS_SPARK_GRADATION",
    "city_registration": "CITY",
    "client_type": "CL_TYPE",
}
_UNKNOWN = {"", "-", "—", "нет данных", "неизвестно", "unknown", "null", "none", "nan", "n/a"}


def _text(value):
    return str(value).strip() if value is not None and str(value).strip() else None


def parse_employee_band(value):
    """Return only bounds explicitly present in source text, never a midpoint.

    A pure integer is retained as a source count with unknown reference period.
    An unrecognized encoding has no numeric interpretation. No undocumented
    numeric sentinel (including zero) is converted to missing data.
    """
    raw = _text(value)
    if raw is None or raw.casefold() in _UNKNOWN:
        return None
    text = raw.casefold().replace("\u00a0", " ").strip()
    text = re.sub(r"\s*(?:чел(?:овек)?\.?|работников|сотрудников)\s*$", "", text).strip()
    text = re.sub(r"\s+", "", text).replace("–", "-").replace("—", "-")
    result = {"lower": None, "upper": None, "lower_inclusive": None,
              "upper_inclusive": None, "unit": "people", "period": None,
              "basis": "source_band"}
    if match := re.fullmatch(r"(?:от)?(\d+)(?:-|до)(\d+)", text):
        lower, upper = map(int, match.groups())
        if lower > upper:
            return None
        result.update(lower=lower, upper=upper, lower_inclusive=True, upper_inclusive=True)
    elif match := re.fullmatch(r"(\d+)", text):
        number = int(match.group(1))
        result.update(lower=number, upper=number, lower_inclusive=True, upper_inclusive=True,
                      basis="source_count")
    elif match := re.fullmatch(r"(>=|>|более|свыше|от)(\d+)", text):
        result.update(lower=int(match.group(2)), lower_inclusive=match.group(1) in {">=", "от"})
    elif match := re.fullmatch(r"(<=|<|менее|до)(\d+)", text):
        # Russian 'до N' alone does not unambiguously establish inclusion of N.
        result.update(upper=int(match.group(2)), upper_inclusive=(True if match.group(1) == "<="
                      else None if match.group(1) == "до" else False))
    elif match := re.fullmatch(r"(\d+)\+", text):
        result.update(lower=int(match.group(1)), lower_inclusive=True)
    else:
        return None
    if any(result[k] is not None and result[k] > 10**12 for k in ("lower", "upper")):
        return None
    return result


def organizer_factors(row: Mapping) -> dict:
    """Canonical raw factors with explicit employee-source reconciliation.

    Different FNS/SPARK bands are not averaged, intersected or promoted into a
    more precise estimate. Original source strings are always retained.
    """
    result = {name: _text(row.get(column)) for name, column in ORGANIZER_FACTOR_COLUMNS.items()}
    result.update(employees_band=None, employees_interval=None, employee_band_status="unknown",
                  factor_quality_flags=[], factor_method=FACTOR_VERSION,
                  factor_limitations=["company_factor_reference_date_unknown",
                      "registration_is_not_operations", "company_age_is_not_business_maturity",
                      "business_stage_label_dictionary_unverified", "negative_mark_semantics_unknown",
                      "client_type_is_not_business_model", "employee_reference_period_unknown"])
    known = [result[name] for name in ("employees_fns_band", "employees_spark_band")
             if result[name] is not None and result[name].casefold() not in _UNKNOWN]
    if not known:
        return result
    parsed = [parse_employee_band(value) for value in known]
    identical = len({value.casefold() for value in known}) == 1
    equivalent = len(known) == 2 and parsed[0] is not None and parsed[0] == parsed[1]
    if len(known) == 2 and not (identical or equivalent):
        result["employee_band_status"] = "conflict"
        result["factor_quality_flags"].append("employees_fns_spark_conflict")
        result["factor_limitations"].append("employee_sources_conflict_no_reconciled_count")
        return result
    result["employees_band"] = known[0]
    result["employees_interval"] = parsed[0]
    result["employee_band_status"] = ("unparsed" if parsed[0] is None else
                                       "single_source" if len(known) == 1 else "consistent")
    if parsed[0] is None:
        result["factor_limitations"].append("employee_band_numeric_bounds_unknown")
    return result
