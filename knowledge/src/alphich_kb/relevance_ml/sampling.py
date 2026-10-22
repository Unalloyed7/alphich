"""Bounded, reproducible company/news pair sampling, with no inferred labels.

OKVED divisions provide coarse sampling strata, not proof that a news event is
applicable. Related-sector and random pairs must still be judged independently.
All company snapshots are private dataset material; aggregate reports omit IDs.
"""

import csv
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from ..common import KnowledgeError, digest, file_hash
from .dataset import instant, validate_pair


SAMPLING_VERSION = "alphich-relevance-sampling-v2-webinar"
# OKVED 2 sections/divisions; deliberately coarser than the detailed activity code.
OKVED_SECTORS = (
    (1, 3, "agriculture"), (5, 9, "mining"), (10, 33, "manufacturing"),
    (35, 35, "energy"), (36, 39, "water_waste"), (41, 43, "construction"),
    (45, 47, "trade"), (49, 53, "transport"), (55, 56, "hospitality"),
    (58, 63, "information"), (64, 66, "finance"), (68, 68, "real_estate"),
    (69, 75, "professional"), (77, 82, "admin"), (84, 84, "public"),
    (85, 85, "education"), (86, 88, "health"), (90, 93, "culture"),
    (94, 96, "services"), (97, 98, "households"), (99, 99, "extraterritorial"),
)
_SECTORS = {item[2] for item in OKVED_SECTORS}
_RELATED = {
    "agriculture": ("manufacturing", "trade", "transport"),
    "mining": ("manufacturing", "energy", "transport"),
    "manufacturing": ("trade", "transport", "energy", "mining"),
    "energy": ("manufacturing", "construction", "transport"),
    "water_waste": ("construction", "manufacturing"),
    "construction": ("real_estate", "manufacturing", "transport"),
    "trade": ("manufacturing", "transport", "agriculture"),
    "transport": ("trade", "manufacturing", "energy"),
    "hospitality": ("trade", "transport", "services"),
    "information": ("professional", "finance", "trade"),
    "finance": ("real_estate", "construction", "trade"),
    "real_estate": ("construction", "finance"),
    "professional": ("information", "construction", "finance"),
    "admin": ("professional", "services"),
    "public": ("education", "health", "construction"),
    "education": ("information", "professional"),
    "health": ("manufacturing", "professional"),
    "culture": ("hospitality", "services"),
    "services": ("trade", "hospitality", "admin"),
}
_TERMS = {
    "agriculture": ("агро", "сельхоз", "сельск", "урожай", "зерн", "фермер"),
    "mining": ("добыч", "рудник", "уголь", "нефтедобы"),
    "manufacturing": ("промышлен", "завод", "производств", "металлург"),
    "energy": ("энергет", "электроэнерг", "электросет", "генерац"),
    "water_waste": ("водоснабж", "водоотвед", "отход", "утилизац"),
    "construction": ("строитель", "застрой", "стройматериал", "ипотек"),
    "trade": ("торгов", "ритейл", "рознич", "оптов", "маркетплейс"),
    "transport": ("транспорт", "перевоз", "логистик", "авиакомпан", "груз"),
    "hospitality": ("гостиниц", "отел", "ресторан", "туризм"),
    "information": ("программ", "цифров", "информационн", "телеком", "it ", "ит-компан"),
    "finance": ("банк", "кредит", "ключев", "страхов", "депозит", "финансов"),
    "real_estate": ("недвижим", "аренд", "офисн"),
    "professional": ("консалт", "аудит", "инжиниринг", "исследован"),
    "admin": ("клининг", "аутсорс", "охранн"),
    "public": ("госуправ", "муниципал", "госзакуп"),
    "education": ("образован", "школ", "университет", "студент"),
    "health": ("медицин", "здравоохран", "больниц", "фармацевт"),
    "culture": ("культур", "театр", "музей", "спортив"),
    "services": ("бытов", "ремонт", "сервисн"),
}
_PRODUCT_COLUMNS = {
    "CL_PRODUCT_RKO_MARK": "rko", "CL_PRODUCT_GAR_MARK": "guarantee",
    "CL_PRODUCT_LOA_MARK": "loan", "CL_PRODUCT_LEA_MARK": "lease",
    "CL_PRODUCT_DEP_MARK": "deposit", "CL_PRODUCT_ZPP_MARK": "payroll", "FACTORING_MARK": "factoring",
}


def okved_sector(code):
    value = str(code).strip() if code is not None else ""
    match = re.match(r"^(\d{1,2})(?:[.,]|$)", value)
    if not match:
        return "unknown"
    division = int(match.group(1))
    return next((sector for lower, upper, sector in OKVED_SECTORS if lower <= division <= upper), "unknown")


def article_sectors(article):
    """Heuristic sampling strata only; never relevance labels or event evidence."""
    topics = article.get("topics", [])
    direct = {value for value in topics if isinstance(value, str) and value in _SECTORS} if isinstance(topics, list) else set()
    if direct:
        return sorted(direct)
    value = (str(article.get("title", "")) + " " + str(article.get("text", ""))[:12000]).casefold().replace("ё", "е")
    scores = {sector: sum(term in value for term in terms) for sector, terms in _TERMS.items()}
    best = max(scores.values(), default=0)
    return sorted(sector for sector, score in scores.items() if score == best and best > 0) or ["unknown"]


def _flag(value):
    if value in {"1", "1.0", "true", "True"}:
        return True
    if value in {"0", "0.0", "false", "False"}:
        return False
    return None


def _company_values(row, issues):
    from alphich.organizer_semantics import SOURCE_REF, revenue_metadata
    result = {}
    columns = {"MAINOKVED_CODE": "okved_code", "MAINOKVED_NAME": "okved_name",
               "REGION": "registration_region"}
    for column, name in columns.items():
        if row.get(column):
            result[name] = (row[column], column)
    revenue = revenue_metadata(row.get("REVENUE_COMPANY"))
    issues.update(revenue["quality_flags"])
    if revenue["amount"] is not None:
        result["revenue_raw"] = (revenue["amount"], "REVENUE_COMPANY")
        for key in ("unit", "currency", "period", "scope"):
            result["revenue_" + key] = (revenue[key], "REVENUE_COMPANY+" + SOURCE_REF)
    for column, name in {**_PRODUCT_COLUMNS, "FTS_FLAG": "fts_flag"}.items():
        if row.get(column):
            flag = _flag(row[column])
            if flag is None:
                issues["unrecognized_boolean"] += 1
            else:
                result[name] = (flag, column)
    countries = sorted({country.strip() for country in row.get("FTS_COUNTRY_NAME", "").replace("|", ",").split(",")
                        if country.strip()})
    fts = result.get("fts_flag", (None,))[0]
    if countries and fts is False:
        issues["fts_country_without_flag"] += 1
        # Keep the contradictory source facts separate; do not infer trade activity.
    if countries:
        result["fts_countries"] = (countries, "FTS_COUNTRY_NAME")
    employee_columns = [column for column in ("AMOUNT_WORKERS_FNS_GRADATION", "AMOUNT_WORKERS_SPARK_GRADATION")
                        if row.get(column)]
    employee_values = {row[column] for column in employee_columns}
    if len(employee_values) == 1:
        result["employees_band"] = (next(iter(employee_values)), "+".join(employee_columns))
    elif len(employee_values) > 1:
        issues["inconsistent_employee_band_omitted"] += 1
    return result


def load_company_snapshots(path):
    """Read organizer CSV locally, quarantining every conflicting version of an ID.

    Flags have the backend importer's strict known encodings. Revenue uses the
    organizer webinar meanings; individual/group scope stays unknown.
    No network, banking interpretation, group-debt inference or data publication.
    """
    path = Path(path)
    if not path.is_file() or path.stat().st_size > 200_000_000:
        raise KnowledgeError("Company CSV must be a local file no larger than 200 MB")
    source_hash = file_hash(path)
    states, conflicts, issues, row_counts = {}, set(), Counter(), Counter()
    source_rows = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        header = stream.readline()
        delimiter = ";" if header.count(";") >= header.count(",") else ","
        stream.seek(0)
        reader = csv.DictReader(stream, delimiter=delimiter)
        if (not reader.fieldnames
                or len({name.strip() for name in reader.fieldnames}) != len(reader.fieldnames)):
            raise KnowledgeError("Company CSV has a missing or duplicate header")
        if not {"SYNTHETIC_U", "MAINOKVED_CODE"}.issubset({name.strip() for name in reader.fieldnames}):
            raise KnowledgeError("Company CSV needs SYNTHETIC_U and MAINOKVED_CODE")
        for row_number, raw in enumerate(reader, start=2):
            source_rows += 1
            row = {key.strip(): (value or "").strip() for key, value in raw.items() if key is not None}
            company_id = row.get("SYNTHETIC_U", "")
            if not company_id or len(company_id) > 160:
                issues["missing_or_invalid_company_id"] += 1
                continue
            row_counts[company_id] += 1
            if None in raw or any(value is None for value in raw.values()):
                conflicts.add(company_id)
                issues["malformed_row"] += 1
                continue
            signature = digest(row)
            if company_id in states:
                if signature != states[company_id][0]:
                    conflicts.add(company_id)
                else:
                    issues["identical_duplicate_rows"] += 1
                continue
            states[company_id] = (signature, row_number, _company_values(row, issues))
    if file_hash(path) != source_hash:
        raise KnowledgeError("Company CSV changed while being read")
    companies, sectors = [], Counter()
    no_facts = 0
    for company_id in sorted(states):
        signature, row_number, values = states[company_id]
        if company_id in conflicts:
            continue
        if not values:
            no_facts += 1
            continue
        profile_version = "csv_" + digest({"source_sha256": source_hash, "row_signature": signature,
                                          "sampling_semantics": SAMPLING_VERSION})
        facts = []
        for name, (value, column) in sorted(values.items()):
            facts.append({"id": "feature_" + digest([company_id, profile_version, name])[:24],
                          "name": name, "value": value,
                          "source_ref": f"csv_sha256:{source_hash}:row:{row_number}:column:{column}"})
        companies.append({"company_id": company_id, "profile_version": profile_version, "facts": facts})
        sectors[okved_sector(values.get("okved_code", (None,))[0])] += 1
    return companies, {"schema_version": SAMPLING_VERSION, "source_sha256": source_hash,
                       "source_rows": source_rows, "unique_company_ids": len(set(states) | conflicts),
                       "usable_companies": len(companies), "quarantined_company_count": len(conflicts),
                       "quarantined_source_rows": sum(row_counts[company_id] for company_id in conflicts),
                       "companies_without_usable_facts": no_facts, "sectors": dict(sorted(sectors.items())),
                       "quality_issue_counts": dict(sorted(issues.items())),
                       "limitations": ["Company geography is registration only unless separately sourced",
                                       "Revenue is RUB for 2025; company/group scope is unknown",
                                       "Current profiles do not establish historical company exposures",
                                       "No group exposure is inferred"]}


def _company_sector(company):
    codes = [fact.get("value") for fact in company.get("facts", [])
             if fact.get("name") == "okved_code" and fact.get("source_ref")]
    return okved_sector(codes[0]) if len(codes) == 1 else "unknown"


def sample_pairs(articles, companies, *, as_of, role="cfo", pairs_per_article=3, limit=30000, seed=42):
    """Pick same-sector, related-sector and random pairs without a Cartesian product.

    The random arm supplies potential cross-sector negatives, not confirmed
    negatives: macroeconomic events may affect almost every sector. Within each
    stratum companies rotate through a seeded shuffled list, rather than always
    selecting its first profile. ``limit`` bounds output, not a paid-label budget.
    """
    if type(pairs_per_article) is not int or not 1 <= pairs_per_article <= 20:
        raise KnowledgeError("pairs_per_article must be 1..20")
    if type(limit) is not int or not 1 <= limit <= 200000:
        raise KnowledgeError("Pair limit must be 1..200000")
    if type(seed) is not int:
        raise KnowledgeError("Sampling seed must be an integer")
    if role not in {"cfo", "ceo"}:
        raise KnowledgeError("Role must be cfo or ceo")
    as_of = instant(as_of).isoformat()
    articles, companies = list(articles), list(companies)
    if len(articles) > 100000 or len(companies) > 200000:
        raise KnowledgeError("Sampling input exceeds 100000 articles or 200000 company profiles")
    if not companies:
        raise KnowledgeError("Sampling needs at least one company snapshot")
    # Fail clearly for invalid shared profile metadata, rather than reporting
    # every article as invalid because the first profile was malformed.
    validate_pair({"company": companies[0], "article": {
        "article_id": "sampling-schema-check", "source_version": "1", "cluster_id": "sampling-schema-check",
        "title": "Schema validation", "text": "Schema validation article.",
        "url": "https://example.invalid/schema-check", "published_at": None}, "as_of": as_of, "role": role})
    company_ids = [company.get("company_id") for company in companies]
    if any(not isinstance(cid, str) or not cid for cid in company_ids) or len(set(company_ids)) != len(companies):
        raise KnowledgeError("Company snapshots need unique nonempty company_id values")
    rng = random.Random(seed)
    buckets = defaultdict(list)
    for index, company in enumerate(companies):
        buckets[_company_sector(company)].append(index)
    for sector in sorted(buckets):
        bucket = buckets[sector]
        bucket.sort(key=lambda index: companies[index]["company_id"])
        rng.shuffle(bucket)
    cursors, sector_use = Counter(), Counter()
    used_companies, pair_kinds, company_sectors = set(), Counter(), Counter()
    result, seen_articles, seen_clusters = [], set(), set()
    invalid_articles, duplicate_articles = 0, 0
    ordered_articles = sorted(articles, key=lambda article: (str(article.get("article_id", "")), str(article.get("source_version", ""))))
    rng.shuffle(ordered_articles)

    def choose(sectors, exclude):
        candidates = sorted({sector for sector in sectors if sector in buckets})
        if not candidates:
            return None
        # Broad random arm balances available sectors; within a sector every
        # eligible company is reachable through a complete rotation.
        rng.shuffle(candidates)
        candidates.sort(key=lambda sector: sector_use[sector])
        for sector in candidates:
            bucket = buckets[sector]
            for _ in range(min(len(bucket), len(exclude) + 1)):
                index = bucket[cursors[sector] % len(bucket)]
                cursors[sector] += 1
                if index not in exclude:
                    sector_use[sector] += 1
                    return index
        return None

    for article in ordered_articles:
        if len(result) >= limit:
            break
        try:
            normalized = validate_pair({"company": companies[0], "article": article, "as_of": as_of, "role": role})["article"]
        except (KnowledgeError, TypeError, AttributeError):
            invalid_articles += 1
            continue
        article_key = (normalized["article_id"], normalized["source_version"])
        text_key = digest(" ".join(normalized["text"].casefold().split()))
        cluster_keys = (normalized["cluster_id"], "text_" + text_key)
        if article_key in seen_articles or any(key in seen_clusters for key in cluster_keys):
            duplicate_articles += 1
            continue
        seen_articles.add(article_key)
        seen_clusters.update(cluster_keys)
        sectors = article_sectors(article)
        related = sorted({candidate for sector in sectors for candidate in _RELATED.get(sector, ())} - set(sectors))
        chosen = set()
        for offset in range(min(pairs_per_article, len(companies))):
            if len(result) >= limit:
                break
            requested, kind = (sectors, "same_sector") if offset == 0 else ((related, "related_sector") if offset == 1
                                                                                         else (list(buckets), "random"))
            index = choose(requested, chosen)
            if index is None:
                index, kind = choose(list(buckets), chosen), "random_fallback"
            if index is None:
                break
            chosen.add(index)
            row = validate_pair({"company": companies[index], "article": normalized, "as_of": as_of, "role": role})
            result.append(row)
            used_companies.add(index)
            pair_kinds[kind] += 1
            company_sectors[_company_sector(companies[index])] += 1
    return result, {"schema_version": SAMPLING_VERSION, "seed": seed, "role": role,
                    "input_articles": len(articles), "usable_unique_articles_visited": len(seen_articles),
                    "invalid_articles_skipped": invalid_articles, "reprints_or_duplicates_skipped": duplicate_articles,
                    "eligible_companies": len(companies), "sampled_company_count": len(used_companies),
                    "pairs": len(result), "pairs_per_article_requested": pairs_per_article, "limit": limit,
                    "pair_kinds": dict(sorted(pair_kinds.items())), "company_sector_pair_counts": dict(sorted(company_sectors.items())),
                    "labels_generated": 0,
                    "limitations": ["Sector/topic matching is a sampling heuristic, never a relevance label",
                                    "Random and related-sector pairs may be relevant; no negative labels are inferred",
                                    "Only supplied reprint clusters and exact normalized copies are deduplicated"]}
