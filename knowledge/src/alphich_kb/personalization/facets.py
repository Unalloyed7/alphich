"""Bounded, evidence-linked search queries; absent products are never query terms."""

from ..common import digest
from . import VERSION
from .inputs import industry_matches

# Small development dictionary, not a complete OKVED taxonomy or exposure model.
SECTOR_TERMS = {
    "41": ["строительство", "жильё", "застройщики"], "42": ["строительство", "инженерные сооружения", "инфраструктура"],
    "24": ["металлургия", "сталь", "прокат"], "25": ["металлоконструкции", "металлоизделия"],
    "26": ["электроника", "микросхемы"], "27": ["электрооборудование", "электротехника"],
    "46.21": ["зерно", "пшеница", "аграрный рынок"], "46.72": ["торговля металлами", "сталь", "металлопродукция"],
    "49": ["грузоперевозки", "транспорт", "топливо"], "52": ["логистика", "склады", "перевозки"],
    "62": ["программное обеспечение", "разработка ПО", "цифровые услуги"],
    "68": ["коммерческая недвижимость", "арендные платежи", "нежилые помещения"],
    "77": ["аренда оборудования", "арендные услуги"],
}


def sector_terms(code):
    matches = [prefix for prefix in SECTOR_TERMS if industry_matches(code, prefix)]
    return SECTOR_TERMS[max(matches, key=len)] if matches else []


def build_facets(company, role):
    facets = []

    def add(kind, text, feature_ids, **metadata):
        identity = {"company_id": company["id"], "context_version": company["context_version"],
                    "role": role, "kind": kind, "text": text, "feature_ids": sorted(set(feature_ids)), "version": VERSION}
        facets.append({"id": f"facet_{digest(identity)}", **identity, **metadata})

    industry = company["industry"]
    label = (industry["name"] or "")[:240]
    if company["bindings"]["industry"] and (label or industry["code"]):
        add("industry", f"{label}. {'; '.join(sector_terms(industry['code']))}. Спрос, издержки и регулирование отрасли.",
            company["bindings"]["industry"])
    positive = [p for p in ("loan", "deposit", "lease", "factoring")
                if company["products"][p] is True and company["bindings"][p]]
    descriptions = {"loan": "Стоимость кредитного финансирования, ключевая ставка",
                    "deposit": "Доходность размещения свободных средств и депозитов",
                    "lease": "Условия лизингового финансирования", "factoring": "Факторинг и оборотное финансирование"}
    if positive:
        add("finance", ". ".join(descriptions[p] for p in positive),
            [i for p in positive for i in company["bindings"][p]], positive_products=positive)
    if company["fts_flag"] is True and company["countries"] and company["bindings"]["fts"]:
        add("trade", f"{label}. Торговые правила, пошлины и международная логистика: {', '.join(company['countries'][:3])}.",
            company["bindings"]["fts"] + company["bindings"]["industry"], countries=company["countries"][:3])
    if company["region"] and company["bindings"]["region"]:
        add("region", f"{str(company['region'])[:120]}. {label}. Регулирование бизнеса и инфраструктура региона регистрации.",
            company["bindings"]["region"] + company["bindings"]["industry"])
    sectors = sorted(company["payment_sectors"], key=lambda s: (-s["share"], s["code"] or ""))[:2]
    if sectors:
        add("payments", "Изменения в отраслях известных платёжных контрагентов: " + "; ".join(str(s["name"])[:140] for s in sectors),
            [s["feature_id"] for s in sectors], payment_sectors=sectors)
    add("role", "Казначейство, стоимость финансирования, ликвидность и оборотный капитал." if role == "cfo"
        else "Спрос, инвестиции, рынок труда и стратегия развития бизнеса.", [])
    add("macro", "Россия. Решения Банка России, инфляция, деловая активность и условия ведения бизнеса.", [])
    return facets
