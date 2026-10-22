"""Documented meanings of the 2026 hackathon organizer fields.

Authority: supplied webinar answers, questions 10, 14–20 and 23. These rules
apply to the organizer dataset, not arbitrary company uploads. No segment or
payment cutoff year is supplied, so neither is inferred from amount or time.
"""
from decimal import Decimal, InvalidOperation

VERSION = "organizer-webinar-semantics-v1"
SOURCE_REF = "organizer-webinar:news-feed:q10,q14-20,q23"
PAYMENT_PERIOD = {"basis": "year_to_date", "start_month_day": "01-01",
                  "year": None, "end": None}


def revenue_metadata(raw):
    value = None
    issue = None
    if raw is not None and str(raw).strip():
        try:
            text = str(raw).strip().replace("\u00a0", "").replace(" ", "").replace(",", ".")
            if len(text) > 128:
                raise ValueError()
            amount = Decimal(text)
            if not amount.is_finite() or amount < 0 or abs(amount.adjusted()) > 40:
                raise ValueError()
            if amount == 1000:
                issue = "revenue_placeholder_1000"
            elif amount == 0:
                issue = "revenue_missing_or_zero"
            else:
                value = format(amount, "f").rstrip("0").rstrip(".") if amount.as_tuple().exponent < 0 else format(amount, "f")
        except (InvalidOperation, ValueError, TypeError):
            issue = "revenue_invalid"
    else:
        issue = "revenue_missing_or_zero"
    return {"amount": value, "unit": "RUB", "currency": "RUB", "period": "2025",
            # The supplied CSV has no business-segment column. Large-business
            # values are group revenue; this cannot be resolved by a threshold.
            "scope": "company_or_group_unknown", "source_ref": SOURCE_REF,
            "quality_flags": [issue] if issue else []}
