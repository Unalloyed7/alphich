"""Thematic applicability rules; never financial-loss or gain estimates."""

import re
from urllib.parse import urlsplit

from .inputs import country, industry_matches, instant, words
from .facets import sector_terms

BUSINESS = {"policy_rate", "inflation", "fuel", "fx", "trade_rules", "logistics", "regulation", "infrastructure",
            "labor", "consumer_demand", "sector", "commodities"}
TOPIC_PATTERNS = {
    "policy_rate": r"ключев\w*\s+ставк\w*|ставк\w*\s+фрс", "inflation": r"инфляц\w*|потребительск\w*\s+цен\w*|цен\w*\s+выросл\w*",
    "fuel": r"бензин\w*|топлив\w*", "trade_rules": r"пошлин\w*|таможенн\w*|экспортн\w*\s+ограничен\w*",
    "logistics": r"логист\w*|грузоперевоз\w*|грузов\w*\s+перевоз\w*", "regulation": r"регулирован\w*|налог\w*|требован\w*",
    "infrastructure": r"инфраструктур\w*|транспортн\w*\s+ограничен\w*", "labor": r"зарплат\w*|ваканси\w*|рын\w*\s+труда",
    "consumer_demand": r"розничн\w*|потребительск\w*\s+спрос\w*|потреблен\w*",
}

# Raw mentions nominate extraction only. Generic infrastructure includes IT
# cabling and cannot by itself establish civil-engineering relevance.
RAW_SECTOR_PATTERNS = {
    "41": r"строительств\w*|ипотек\w*|жиль[яеёю]\w*|жилищн\w*|новострой\w*|застройщик\w*",
    "46.51": r"компьютер\w*|ноутбук\w*|комплектующ\w*|\bDRAM\b|\bNAND\b|memory\s+(?:market|pric\w*)|notebook\w*|\bPC\s+market",
    "42": r"инженерн\w*\s+сооружен\w*|транспортн\w*\s+инфраструктур\w*|"
          r"строительств\w*[^.!?]{0,80}(?:дорог|мост|тоннел|магистрал)|реконструкц\w*|благоустройств\w*|реставрац\w*",
    "62": r"программн\w*\s+обеспечен\w*|программист\w*|разработчик\w*\s+по\b|"
          r"разработк\w*\s+(?:по\b|приложен\w*)|свободн\w*\s+по\b|"
          r"операционн\w*\s+систем\w*|astra\s+linux|vr[- ](?:проект|обучен)\w*",
}

EXPLANATIONS = {
    "credit_rate_channel": "В профиле указан кредитный продукт. Событие о ключевой ставке относится к условиям финансирования; влияние зависит от договора.",
    "deposit_rate_channel": "В профиле указан депозитный продукт. Событие о ставке относится к условиям размещения средств; остатки и условия неизвестны.",
    "lease_rate_channel": "В профиле указан лизинговый продукт. Применимость изменения условий финансирования зависит от договора.",
    "national_rate_context": "Событие относится к денежно-кредитным условиям российского рынка; индивидуальная финансовая экспозиция не установлена.",
    "national_price_context": "Событие описывает общий инфляционный контекст; состав затрат компании неизвестен.",
    "national_labor_context": "Событие описывает общий контекст рынка труда; условия найма компании неизвестны.",
    "national_demand_context": "Событие описывает общий контекст спроса; клиентская база компании неизвестна.",
    "national_fx_context": "Событие относится к валютному контексту рынка; валюты договоров и позиция компании неизвестны.",
    "industry_scope_match": "Отрасль в проверенном факте совпадает с ОКВЭД компании. Денежный эффект этим совпадением не установлен.",
    "country_trade_channel": "Страна из сведений ФТС совпадает со страной торгового изменения. Направление торговли и конкретные сделки неизвестны.",
    "registration_region_context": "Регион события совпадает с регионом регистрации. Расположение операций компании требует уточнения.",
    "payment_neighbor_sector": "Событие относится к отрасли известных платёжных контрагентов. Хозяйственный смысл связи и период платежей неизвестны.",
    "unverified_macro_mention": "В тексте упоминается макроэкономическая тема российского рынка; необходимо извлечь и проверить факты.",
    "unverified_credit_rate_channel": "В тексте упоминается ставка, а в профиле указан кредит. Связь предварительная и требует разбора публикации.",
    "unverified_deposit_rate_channel": "В тексте упоминается ставка, а в профиле указан депозит. Связь предварительная и требует разбора публикации.",
    "unverified_industry_mention": "В тексте упоминается тема отрасли компании; необходимо проверить факты и область применения.",
    "unverified_trade_mention": "В тексте есть торговая тема и страна из сведений ФТС; необходимо проверить факты и направление торговли.",
}


def signal(event, pattern):
    for source in event["sources"]:
        found = re.search(pattern, source["text"], re.IGNORECASE)
        if found:
            return {"source_id": source["id"], "source_version": source["version"],
                    "start": found.start(), "end": found.end(), "text": found.group()}
    found = re.search(pattern, event["title"], re.IGNORECASE)
    if found:
        return {"event_id": event["id"], "event_version": event["version"], "field": "title",
                "start": found.start(), "end": found.end(), "text": found.group()}
    return None


def match_reasons(company, event, scope):
    reasons = []
    facts = [f for f in event["facts"] if instant(f["source_available_at"]) <= instant(scope.as_of)]
    topics = set()

    def add(code, tier, base, feature_ids=(), fact_ids=(), angle=None, limitations=(), source_refs=(), **extra):
        reasons.append({"code": code, "tier": tier, "base": base, "feature_ids": sorted(set(feature_ids)),
                        "fact_ids": sorted(set(fact_ids)), "angle": angle or code, "limitations": list(limitations),
                        "explanation": EXPLANATIONS[code], "source_refs": list(source_refs), **extra})

    for fact in facts:
        attrs = fact["attributes"]
        ft = set(attrs["topics"])
        topics.update(ft)
        local = attrs.get("geography") == scope.market_country
        sector_scope = local or attrs.get("geography") == "GLOBAL"
        fact_ids = [fact["id"]]
        if "policy_rate" in ft and local:
            add("national_rate_context", "context", 57, fact_ids=fact_ids, angle="monetary_conditions")
            for product, code, base, angle, limits in [
                ("loan", "credit_rate_channel", 88, "financing_cost", ["credit_balance_unknown", "credit_rate_type_unknown", "repricing_dates_unknown"]),
                ("deposit", "deposit_rate_channel", 86, "cash_placement_yield", ["deposit_balance_unknown", "deposit_term_and_rate_unknown"]),
                ("lease", "lease_rate_channel", 80, "lease_financing", ["lease_contract_terms_unknown"]),
            ]:
                if company["products"][product] is True and company["bindings"][product]:
                    add(code, "direct", base, company["bindings"][product], fact_ids, angle, limits)
        for topic, base, code in [("inflation", 50, "national_price_context"), ("labor", 48, "national_labor_context"),
                                  ("consumer_demand", 49, "national_demand_context"), ("fx", 42, "national_fx_context")]:
            if topic in ft and local:
                add(code, "context", base, fact_ids=fact_ids, limitations=["company_exposure_not_quantified"])
        sector_match = any(industry_matches(company["industry"]["code"], prefix) for prefix in attrs["industry_codes"])
        if sector_match and company["bindings"]["industry"] and ft & BUSINESS:
            add("industry_scope_match", "direct" if sector_scope else "hypothesis", 82 if sector_scope else 30,
                company["bindings"]["industry"], fact_ids, "industry_conditions",
                ["thematic_applicability_not_financial_impact"] if sector_scope else ["foreign_operational_presence_unknown"])
        matched_countries = set(company["countries"]) & {country(c) for c in attrs["countries"]}
        if (matched_countries and company["fts_flag"] is True and company["bindings"]["fts"] and ft & {"trade_rules", "logistics"}
                and (not attrs["industry_codes"] or sector_match)):
            add("country_trade_channel", "context", 65, company["bindings"]["fts"], fact_ids, "cross_border_conditions",
                ["trade_direction_unknown", "transaction_currency_unknown"], countries=sorted(matched_countries))
        if company["region"] and company["bindings"]["region"] and ft & {"regulation", "infrastructure"}:
            if (words(company["region"]) in {words(region) for region in attrs["regions"]} and local
                    and (not attrs["industry_codes"] or sector_match)):
                add("registration_region_context", "context", 46, company["bindings"]["region"], fact_ids,
                    "regional_conditions", ["registration_is_not_operational_geography"])
        if sector_scope and ft & BUSINESS:
            for sector in company["payment_sectors"]:
                if any(industry_matches(sector["code"], prefix) for prefix in attrs["industry_codes"]):
                    add("payment_neighbor_sector", "context", 40 + 8 * sector["share"], [sector["feature_id"]], fact_ids,
                        "counterparty_sector_context", sector["limitations"] + ["payment_link_business_meaning_unknown"],
                        coverage=sector["coverage"], share_of_all_flow=sector["share"], direction=sector["direction"])
    if reasons:
        return reasons, topics
    if facts and topics and not topics & BUSINESS:
        return [], topics
    raw_topics = {topic for topic, pattern in TOPIC_PATTERNS.items() if signal(event, pattern)}
    combined = words(event["title"] + " " + event["text"])
    local_signal = bool(re.search(r"банк россии|цб рф|росси\w*", combined)) or any(
        urlsplit(s.get("url") or "").hostname in {"cbr.ru", "www.cbr.ru"} for s in event["sources"])
    if local_signal and raw_topics & {"policy_rate", "inflation", "labor", "consumer_demand"}:
        refs = [signal(event, TOPIC_PATTERNS[t]) for t in sorted(raw_topics & {"policy_rate", "inflation", "labor", "consumer_demand"})]
        add("unverified_macro_mention", "hypothesis", 36, source_refs=refs,
            limitations=["news_facts_not_verified", "requires_extraction"])
        if "policy_rate" in raw_topics:
            for product, code, angle in [("loan", "unverified_credit_rate_channel", "financing_cost"),
                                         ("deposit", "unverified_deposit_rate_channel", "cash_placement_yield")]:
                if company["products"][product] is True and company["bindings"][product]:
                    add(code, "hypothesis", 38, company["bindings"][product], angle=angle, source_refs=refs,
                        limitations=["news_facts_not_verified", "contract_terms_unknown", "requires_extraction"])
    if company["bindings"]["industry"]:
        terms = sector_terms(company["industry"]["code"])
        specific = next((pattern for prefix,pattern in RAW_SECTOR_PATTERNS.items()
                         if industry_matches(company["industry"]["code"],prefix)), None)
        patterns = [specific] if specific else [re.escape(term[:-1] if len(term)>5 else term)+r"\w*" for term in terms]
        refs = [signal(event,pattern) for pattern in patterns]
        refs = [ref for ref in refs if ref]
        if refs and (not facts or topics & BUSINESS):
            title_match = any(re.search(pattern,event["title"],re.I) for pattern in patterns)
            add("unverified_industry_mention", "hypothesis", 39 if title_match else 33, company["bindings"]["industry"], source_refs=refs,
                limitations=["news_facts_not_verified", "requires_extraction"])
    if company["fts_flag"] is True and company["bindings"]["fts"] and raw_topics & {"trade_rules", "logistics"}:
        matches = [c for c in company["countries"] if c in combined]
        if matches:
            add("unverified_trade_mention", "hypothesis", 34, company["bindings"]["fts"],
                limitations=["news_facts_not_verified", "trade_direction_unknown", "requires_extraction"])
    return reasons, topics or raw_topics
