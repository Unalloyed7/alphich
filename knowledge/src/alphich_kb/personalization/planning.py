"""Generate bounded bank-search requests from relevance reasons, not bank availability."""

from datetime import timedelta

from ..common import digest
from .inputs import instant


def bank_questions(company, event, reasons, topics, scope):
    codes = {r["code"] for r in reasons}
    question_specs = []
    if "policy_rate" in topics or codes & {"credit_rate_channel", "deposit_rate_channel", "lease_rate_channel", "national_rate_context"}:
        horizons = sorted({f["attributes"]["forecast_horizon"] for f in event["facts"]
                           if "policy_rate" in f["attributes"]["topics"] and f["attributes"].get("forecast_horizon")
                           and instant(f["source_available_at"]) <= instant(scope.as_of)})
        horizon = horizons[0] if len(horizons) == 1 else f"{instant(scope.as_of).year}-12-31"
        question_specs.append(("forecast", "Какую траекторию ключевой ставки описывает банк и какие условия могут изменить прогноз?",
                               "policy_rate", horizon))
        if codes & {"credit_rate_channel", "unverified_credit_rate_channel"}:
            question_specs.append(("mechanism", "Как денежно-кредитная политика связана со стоимостью финансирования компаний и условиями пересмотра кредитной ставки?", None, None))
        elif codes & {"deposit_rate_channel", "unverified_deposit_rate_channel"}:
            question_specs.append(("mechanism", "Какие факторы денежно-кредитной политики влияют на доходность размещения свободных средств и условия депозитов?", None, None))
    elif "fuel" in topics:
        question_specs.append(("interpretation", "Как банк оценивает влияние цен на топливо на инфляционные риски?", "fuel_inflation_risk", None))
    elif "inflation" in topics:
        question_specs.append(("forecast", "Какой прогноз потребительской инфляции даёт банк и каковы условия прогноза?",
                               "consumer_inflation_yoy", f"{instant(scope.as_of).year}-12-31"))
    if not question_specs:
        label = str(company["industry"]["name"] or "бизнес")[:180]
        question_specs.append(("sector_context", f"{label}. Какие механизмы спроса, издержек или регулирования рассматривает банк по теме: {event['title'][:180]}?", None, None))
    feature_ids = sorted({i for r in reasons for i in r["feature_ids"]})
    fact_ids = sorted({i for r in reasons for i in r["fact_ids"]})
    # PDF publication precision is a day: conservatively use completed UTC days.
    bank_as_of = (instant(scope.as_of).date() - timedelta(days=1)).isoformat()
    result = []
    for purpose, query, metric, horizon in question_specs[:2]:
        request = {"purpose": purpose, "query": query, "metric": metric, "horizon": horizon,
                   "as_of": bank_as_of, "event_id": event["id"], "event_version": event["version"],
                   "feature_ids": feature_ids, "fact_ids": fact_ids, "date_policy": "completed_utc_day",
                   "limitations": ["bank_publication_intraday_time_unknown"]}
        result.append({"id": f"bank_question_{digest(request)}", **request})
    return result
