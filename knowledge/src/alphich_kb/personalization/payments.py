"""Attach audited ETL payment features to a matching API profile, without re-IDing evidence."""
from copy import deepcopy
from decimal import Decimal, InvalidOperation
import math

from ..common import KnowledgeError
from .profiles import normalize_company

METHOD = "nonself_all_flow_denominator_normalized_duplicates_once"


def validate_payment_feature(feature):
    try:
        if feature["method"] != METHOD or not feature["source_refs"]:
            raise ValueError()
        value = feature["value"]
        amount = Decimal(value["amount_raw"])
        denominator = Decimal(value["shares_denominator_raw"])
        if not amount.is_finite() or amount < 0 or amount != denominator:
            raise ValueError()
        for key in ("payment_count", "counterparty_count", "unknown_profile_counterparty_count", "top_n"):
            if type(value[key]) is not int or value[key] < 0:
                raise ValueError()
        if not 1 <= value["top_n"] <= 100 or value["unknown_profile_counterparty_count"] > value["counterparty_count"]:
            raise ValueError()
        for key in ("known_profile_amount_share", "within_known_group_amount_share", "top_counterparty_share"):
            share = value.get(key)
            if share is not None and (type(share) not in (int,float) or not math.isfinite(share) or not 0 <= share <= 1):
                raise ValueError()
        for key in ("top_counterparties", "known_sectors"):
            if not isinstance(value[key], list) or len(value[key]) > value["top_n"]:
                raise ValueError()
            for item in value[key]:
                subtotal = Decimal(item["amount_raw"])
                share = item["share_of_all_flow"]
                if not subtotal.is_finite() or not 0 <= subtotal <= amount:
                    raise ValueError()
                expected = float(subtotal/amount) if amount else None
                if expected is None:
                    if share is not None: raise ValueError()
                elif type(share) not in (int,float) or not math.isfinite(share) or abs(share-expected) > 1e-9:
                    raise ValueError()
        limitations = set(value["limitations"])
        if value.get("currency") is None and value.get("period") is None:
            # Old immutable exports retain their original unknown semantics.
            if not {"payment_period_unknown", "payment_currency_unknown", "payment_purpose_unknown"} <= limitations:
                raise ValueError()
        elif (value.get("currency") != "RUB"
              or value.get("period") != {"basis": "year_to_date", "start_month_day": "01-01", "year": None, "end": None}
              or value.get("semantics_source_ref") != "organizer-webinar:news-feed:q10,q14-20,q23"
              or not {"payment_period_end_and_year_unknown", "payment_purpose_unknown"} <= limitations):
            raise ValueError()
    except (KeyError, TypeError, ValueError, InvalidOperation, OverflowError):
        raise KnowledgeError("Invalid audited payment feature") from None


def attach_payment_context(profile, context):
    base, prepared = normalize_company(profile), normalize_company(context)
    if base["id"] != prepared["id"] or any(base[k] != prepared[k] for k in ("industry", "region", "products", "fts_flag", "countries")):
        raise KnowledgeError("Prepared payment context disagrees with current company profile")
    result = deepcopy(profile)
    additions = []
    for feature in context["features"]:
        if feature["name"] not in {"payment_incoming", "payment_outgoing"}:
            continue
        validate_payment_feature(feature)
        bound = deepcopy(feature)
        bound.update(bound_profile_version=profile["profile_version"], source_context_version=context["context_version"])
        additions.append(bound)
    if len(additions) != 2 or len({f["name"] for f in additions}) != 2:
        raise KnowledgeError("Both immutable payment directions are required")
    result["features"] = [f for f in result.get("features", []) if f["name"] not in {"payment_incoming", "payment_outgoing"}] + additions
    result["payment_context_version"] = context["context_version"]
    normalize_company(result)
    return result
