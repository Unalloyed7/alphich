"""Company-conditioned E5 features, with no labels or teacher explanations as inputs.

The canonical pair's IDs are used by the dataset layer, never as learned features.
Only allowlisted, source-backed company facts enter the company representation.
Region mentions below are lexical features, not verified event applicability.
Revenue scale is numeric only with explicit RUB/2025 metadata. Company/group
scope remains explicit. Publication age is excluded from the business-fit model.
"""

import json
import math
import re
from decimal import Decimal, InvalidOperation
from datetime import datetime, timezone

import numpy as np

from ..common import KnowledgeError


FEATURE_VERSION = "relevance-features-v2-business-fit"
PRODUCTS = ("loan", "deposit", "lease", "guarantee", "factoring", "rko", "payroll")
_ALIASES = {
    "okved": "okved_code", "industry_code": "okved_code", "industry_name": "okved_name",
    "region": "registration_region", "customs_countries": "fts_countries",
    "countries": "fts_countries", "revenue": "revenue_raw",
    **{name + "_flag": name for name in PRODUCTS},
}
_FACT_NAMES = frozenset({"industry", "okved_code", "okved_name", "registration_region",
                         "operating_regions", "fts", "fts_countries", "fts_flag", "revenue_raw",
                         "revenue_unit", "revenue_currency", "revenue_period", "revenue_scope", "employees_band",
                         "business_goals", "products", *PRODUCTS})
_SCALARS = (
    "cosine_pooled", "cosine_chunk_max", "cosine_chunk_min", "cosine_chunk_std",
    "industry_known", "registration_region_known", "operating_regions_known",
    "operating_region_mentioned", "registration_region_mentioned",
    "trade_countries_known", "trade_country_mentioned", "fts_known", "fts_positive",
    "loan_known", "loan_positive", "deposit_known", "deposit_positive",
    "lease_known", "lease_positive", "guarantee_known", "guarantee_positive",
    "factoring_known", "factoring_positive", "rko_known", "rko_positive", "payroll_known", "payroll_positive",
    "raw_revenue_present", "revenue_unit_known", "revenue_currency_known", "revenue_period_known",
    "reported_revenue_rub_2025_log1p", "reported_revenue_scale_known", "revenue_scope_known",
    "employees_band_known", "business_goals_known", "company_fact_count_log1p",
    "role_cfo", "role_ceo",
    "article_characters_log1p", "article_chunks_log1p", "company_chunks_log1p",
    "company_news_token_jaccard",
)


def _text(value):
    return value.strip() if isinstance(value, str) else ""


def _list_of_text(value):
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return sorted({_text(item) for item in value if _text(item)})


def _normal(value):
    return " ".join(re.findall(r"\w+", value.casefold().replace("ё", "е")))


def _mentioned(values, article):
    normalized = " " + _normal(article) + " "
    return float(any(len(_normal(value)) >= 3 and " " + _normal(value) + " " in normalized
                     for value in values))


def _boolean(value):
    # Strings such as "false" and numbers are deliberately not coerced to truth.
    return value if isinstance(value, bool) else None


def _instant(value, name):
    if not isinstance(value, str):
        raise KnowledgeError(f"{name} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise KnowledgeError(f"{name} must be a timezone-aware ISO timestamp") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise KnowledgeError(f"{name} must include a timezone")
    return parsed.astimezone(timezone.utc)


def _profile(company):
    """Extract only independently source-backed, typed values; ignore all other keys."""
    if not isinstance(company, dict) or not isinstance(company.get("facts"), list):
        raise KnowledgeError("Feature inputs require company.facts")
    entries = []
    for fact in company["facts"]:
        if not isinstance(fact, dict) or not _text(fact.get("id")) or not _text(fact.get("source_ref")):
            continue
        raw_name = fact.get("name")
        name = _ALIASES.get(raw_name, raw_name) if isinstance(raw_name, str) else None
        if isinstance(name, str) and name in _FACT_NAMES:
            entries.append((name, fact.get("value")))
    # Fact order is not a learned signal. Conflicting values are represented as
    # unknown below, rather than silently taking the last record.
    values = {}
    for name, value in entries:
        values.setdefault(name, []).append(value)
    for value in values.get("fts", []):
        if isinstance(value, dict):
            values.setdefault("fts_flag", []).append(value.get("flag"))
            values.setdefault("fts_countries", []).append(value.get("countries"))

    def unique(name, convert):
        converted = [convert(value) for value in values.get(name, [])]
        valid = [value for value in converted if value is not None and value != "" and value != []]
        if not valid:
            return None
        canonical = {json.dumps(value, ensure_ascii=False, sort_keys=True) for value in valid}
        return valid[0] if len(canonical) == 1 else None

    code = unique("okved_code", _text)
    name = unique("okved_name", _text)
    industries = values.get("industry", [])
    if industries:
        industry_codes = {_text(value.get("code")) for value in industries if isinstance(value, dict)} - {""}
        industry_names = {_text(value.get("name")) for value in industries if isinstance(value, dict)} - {""}
        if industry_codes:
            code = next(iter(industry_codes)) if len(industry_codes) == 1 and (not code or code in industry_codes) else None
        if industry_names:
            name = next(iter(industry_names)) if len(industry_names) == 1 and (not name or name in industry_names) else None
    if code and not re.fullmatch(r"\d{2}(?:\.\d{1,2}){0,3}", code):
        code = None
    registration = unique("registration_region", _text)
    regions = unique("operating_regions", _list_of_text) or []
    countries = unique("fts_countries", _list_of_text) or []
    fts = unique("fts_flag", _boolean)
    products = {}
    for product in PRODUCTS:
        candidates = [_boolean(value) for value in values.get(product, [])]
        candidates += [_boolean(value.get(product)) for value in values.get("products", []) if isinstance(value, dict)]
        distinct = {value for value in candidates if value is not None}
        products[product] = next(iter(distinct)) if len(distinct) == 1 else None
    amounts = set()
    for value in values.get("revenue_raw", []):
        try:
            amount = Decimal(str(value).strip().replace(" ", "").replace("\u00a0", "").replace(",", "."))
            if not isinstance(value, bool) and amount.is_finite() and amount > 0 and amount != 1000 and abs(amount.adjusted()) <= 40:
                amounts.add(amount)
        except (InvalidOperation, ValueError):
            pass
    raw_revenue = len(amounts) == 1
    revenue_metadata = {name: unique(name, _text) for name in ("revenue_unit", "revenue_currency", "revenue_period")}
    scaled_revenue = (float(next(iter(amounts))) if raw_revenue
                      and revenue_metadata == {"revenue_unit": "RUB", "revenue_currency": "RUB", "revenue_period": "2025"}
                      else None)
    revenue_scope = unique("revenue_scope", _text)
    employees = unique("employees_band", _text)
    goals = unique("business_goals", _list_of_text) or []
    lines = []
    if code or name:
        lines.append("Отрасль ОКВЭД: " + " ".join(item for item in (code, name) if item))
    if registration:
        lines.append("Регион регистрации (территория операций не подтверждена этим признаком): " + registration)
    if regions:
        lines.append("Подтверждённые регионы операций: " + "; ".join(regions))
    for product, value in products.items():
        if value is not None:
            lines.append(f"Подтверждённый продукт {product}: {'да' if value else 'нет'}; остаток и условия неизвестны")
    if fts is not None:
        lines.append("Признак внешней торговли: " + ("да" if fts else "нет"))
    if countries:
        lines.append("Страны внешнеторговых записей: " + "; ".join(countries))
    if employees:
        lines.append("Подтверждённый диапазон численности сотрудников: " + employees)
    if goals:
        lines.append("Цели бизнеса, указанные в подтверждённом профиле: " + "; ".join(goals))
    # Raw revenue amounts, company/group IDs, facts' IDs and source refs never
    # enter an embedding. Availability masks below preserve their limitations.
    return {
        "text": "\n".join(lines) or "Подтверждённые сведения о деятельности компании отсутствуют.",
        "industry": bool(code or name), "registration": registration, "regions": regions,
        "countries": countries, "fts": fts, "products": products,
        "revenue": bool(raw_revenue), "revenue_metadata": revenue_metadata,
        "scaled_revenue": scaled_revenue, "revenue_scope": revenue_scope,
        "employees": bool(employees), "goals": bool(goals), "fact_count": len(entries),
    }


class FeatureBuilder:
    """Build deterministic pair features with complete, token-budgeted text coverage.

    ``encoder`` supplies dimension, space_id, max_tokens, count_tokens(text),
    encode_query(raw_text), and optionally encode_passages(prefixed_texts).
    E5's symmetric query/query strategy is the default for linear probing;
    asymmetric retrieval is an explicit, manifest-bound alternative.
    """

    def __init__(self, encoder, *, max_article_chars=200_000, max_company_chars=32_000,
                 max_title_chars=2_000, text_strategy="symmetric_query", batch_size=4):
        if text_strategy not in {"symmetric_query", "asymmetric_retrieval"}:
            raise KnowledgeError("Unknown E5 feature text strategy")
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
               for value in (max_article_chars, max_company_chars, max_title_chars, batch_size)) or batch_size > 32:
            raise KnowledgeError("Feature bounds must be positive integers; batch_size must be 1..32")
        self.encoder = encoder
        self.embedding_dimension = int(encoder.dimension)
        if self.embedding_dimension < 1 or not _text(encoder.space_id):
            raise KnowledgeError("A named, non-empty embedding space is required")
        self.max_article_chars = max_article_chars
        self.max_company_chars = max_company_chars
        self.max_title_chars = max_title_chars
        self.text_strategy = text_strategy
        self.batch_size = batch_size
        self.feature_names = tuple(f"{kind}_{index}" for kind in ("company_e5", "article_e5", "abs_diff", "product")
                                   for index in range(self.embedding_dimension)) + _SCALARS

    @property
    def manifest(self):
        return {
            "feature_version": FEATURE_VERSION,
            "encoder_space_id": self.encoder.space_id,
            "embedding_dimension": self.embedding_dimension,
            "dimension": len(self.feature_names),
            "feature_names": list(self.feature_names),
            "configuration": {
                "text_strategy": self.text_strategy, "company_prefix": "query: ",
                "article_prefix": "query: " if self.text_strategy == "symmetric_query" else "passage: ",
                "chunking": "lossless_character_boundaries_token_counted_v1",
                "pooling": "content_token_weighted_mean_then_l2",
                "max_article_chars": self.max_article_chars, "max_company_chars": self.max_company_chars,
                "max_title_chars": self.max_title_chars, "encoder_max_tokens": self.encoder.max_tokens,
                "company_fact_names": sorted(_FACT_NAMES), "company_fact_aliases": dict(_ALIASES),
                "missing_or_conflicting_facts": "unknown", "raw_revenue": "log1p_only_explicit_RUB_2025_reported_scope",
                "target": "business_relevance", "publication_age": "excluded_separate_temporal_assessment",
                "article_attributes": "none_inferred_from_keywords",
                "ids_and_source_refs": "excluded_except_source_presence_validation",
            },
        }

    def _chunks(self, text, prefix):
        chunks = []
        position = 0
        while position < len(text):
            remaining = len(text) - position
            # A character can expand into several tokens; test exact prepared
            # strings instead of assuming a character/token conversion ratio.
            probe = min(remaining, max(32, self.encoder.max_tokens * 4))
            while probe < remaining and self.encoder.count_tokens(prefix + text[position:position + probe]) <= self.encoder.max_tokens:
                probe = min(remaining, probe * 2)
            if probe == remaining and self.encoder.count_tokens(prefix + text[position:]) <= self.encoder.max_tokens:
                end = len(text)
            else:
                low, high, best = 1, probe, 0
                while low <= high:
                    width = (low + high) // 2
                    if self.encoder.count_tokens(prefix + text[position:position + width]) <= self.encoder.max_tokens:
                        best, low = width, width + 1
                    else:
                        high = width - 1
                if best == 0:
                    raise KnowledgeError("Encoder token budget cannot fit one content character")
                end = position + best
                boundary = max(text.rfind(" ", position + best // 2, end), text.rfind("\n", position + best // 2, end))
                if boundary >= position:
                    end = boundary + 1
            chunk = text[position:end]
            if chunk.strip():
                chunks.append(chunk)
            # Whitespace-only portions have no E5 semantic tokens. Their source
            # characters are consumed explicitly; no business text is omitted.
            position = end
        if not chunks:
            raise KnowledgeError("Cannot embed an empty text")
        return chunks

    def _vectors(self, chunks, *, article=False):
        if article and self.text_strategy == "asymmetric_retrieval":
            vectors = self.encoder.encode_passages(["passage: " + chunk for chunk in chunks], batch_size=self.batch_size)
        else:
            # E5Encoder caches query inference; repeated company profiles are
            # reused without storing texts or embeddings on disk.
            vectors = np.asarray([self.encoder.encode_query(chunk) for chunk in chunks], dtype=np.float32)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(chunks), self.embedding_dimension) or not np.isfinite(vectors).all():
            raise KnowledgeError("Encoder returned invalid relevance feature vectors")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0):
            raise KnowledgeError("Encoder returned a zero relevance feature vector")
        return vectors / norms[:, None]

    def _pool(self, chunks, vectors):
        # count_tokens(content) includes a fixed special-token overhead, removed
        # by the count for empty content. Prefix tokens do not receive weight.
        overhead = self.encoder.count_tokens("")
        weights = np.asarray([max(1, self.encoder.count_tokens(chunk) - overhead) for chunk in chunks], dtype=np.float64)
        pooled = np.average(vectors, axis=0, weights=weights)
        norm = np.linalg.norm(pooled)
        if not np.isfinite(norm) or norm <= 0:
            raise KnowledgeError("Cannot pool cancelling or invalid relevance vectors")
        return np.asarray(pooled / norm, dtype=np.float32), weights

    def _row(self, row):
        if not isinstance(row, dict) or not isinstance(row.get("article"), dict):
            raise KnowledgeError("Features require canonical company-news pairs")
        profile = _profile(row.get("company"))
        article = row["article"]
        title, body = article.get("title"), article.get("text")
        if not isinstance(title, str) or not title.strip() or not isinstance(body, str) or not body.strip():
            raise KnowledgeError("A full article title and text are required")
        if len(body) > self.max_article_chars or len(title) > self.max_title_chars:
            raise KnowledgeError("Article exceeds feature character limits; truncation is forbidden")
        if len(profile["text"]) > self.max_company_chars:
            raise KnowledgeError("Company context exceeds feature character limit; truncation is forbidden")
        role = row.get("role")
        if role not in {"cfo", "ceo"}:
            raise KnowledgeError("Feature role must be cfo or ceo")
        as_of = _instant(row.get("as_of"), "as_of")
        publication = article.get("published_at")
        age = 0.0
        if publication is not None:
            age = (as_of - _instant(publication, "published_at")).total_seconds() / 86_400
            if age < 0:
                raise KnowledgeError("A future article cannot enter relevance feature extraction")
        article_text = title + "\n\n" + body
        company_chunks = self._chunks(profile["text"], "query: ")
        article_prefix = "query: " if self.text_strategy == "symmetric_query" else "passage: "
        article_chunks = self._chunks(article_text, article_prefix)
        u, _ = self._pool(company_chunks, self._vectors(company_chunks))
        article_vectors = self._vectors(article_chunks, article=True)
        v, weights = self._pool(article_chunks, article_vectors)
        similarities = np.clip(article_vectors @ u, -1.0, 1.0)
        average = np.average(similarities, weights=weights)
        company_words, article_words = set(_normal(profile["text"]).split()), set(_normal(article_text).split())
        union = company_words | article_words
        scalars = [
            float(np.clip(u @ v, -1, 1)), float(similarities.max()), float(similarities.min()),
            math.sqrt(float(np.average((similarities - average) ** 2, weights=weights))),
            float(profile["industry"]), float(bool(profile["registration"])), float(bool(profile["regions"])),
            _mentioned(profile["regions"], article_text), _mentioned([profile["registration"]] if profile["registration"] else [], article_text),
            float(bool(profile["countries"])), _mentioned(profile["countries"], article_text),
            float(profile["fts"] is not None), float(profile["fts"] is True),
            *[value for product in PRODUCTS for value in (float(profile["products"][product] is not None), float(profile["products"][product] is True))],
            float(profile["revenue"]), *[float(bool(profile["revenue_metadata"][key])) for key in ("revenue_unit", "revenue_currency", "revenue_period")],
            math.log1p(profile["scaled_revenue"]) if profile["scaled_revenue"] is not None else 0.0,
            float(profile["scaled_revenue"] is not None), float(profile["revenue_scope"] in {"company", "group"}),
            float(profile["employees"]), float(profile["goals"]), math.log1p(profile["fact_count"]),
            float(role == "cfo"), float(role == "ceo"),
            math.log1p(len(article_text)), math.log1p(len(article_chunks)), math.log1p(len(company_chunks)),
            len(company_words & article_words) / len(union) if union else 0.0,
        ]
        vector = np.concatenate((u, v, np.abs(u - v), u * v, np.asarray(scalars, dtype=np.float32)))
        if vector.shape != (len(self.feature_names),) or not np.isfinite(vector).all():
            raise KnowledgeError("Relevance feature dimensions or finite-value checks failed")
        return vector

    def transform(self, rows):
        """Return a float32 matrix; label and arbitrary metadata keys are ignored."""
        features = [self._row(row) for row in rows]
        return np.ascontiguousarray(features, dtype=np.float32) if features else np.empty((0, len(self.feature_names)), dtype=np.float32)
