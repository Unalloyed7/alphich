"""Adapt both existing company contracts without fabricating feature evidence."""

import copy
import math
import re

from ..common import KnowledgeError, digest
from .inputs import country

PRODUCTS = {"loan": "CL_PRODUCT_LOA_MARK", "deposit": "CL_PRODUCT_DEP_MARK", "lease": "CL_PRODUCT_LEA_MARK",
            "guarantee": "CL_PRODUCT_GAR_MARK", "factoring": "FACTORING_MARK", "rko": "CL_PRODUCT_RKO_MARK",
            "payroll": "CL_PRODUCT_ZPP_MARK"}


def normalize_company(row):
    profile = row.get("profile", row)
    company_id = row.get("company_id", profile.get("id"))
    if not company_id or (profile.get("id") and company_id != profile["id"]) or not row.get("profile_version"):
        raise KnowledgeError("Company ID and immutable profile_version are required")
    reference_version = row.get("context_version", row["profile_version"])
    industry = profile.get("industry", {})
    code = industry.get("code", industry.get("okved_code"))
    name = industry.get("name", industry.get("okved_name"))
    if code is not None and (not isinstance(code, str) or not re.fullmatch(r"\d{2}(?:\.\d{1,2}){0,3}", code)):
        raise KnowledgeError("Industry code must be an OKVED string or null")
    if name is not None and not isinstance(name, str):
        raise KnowledgeError("Industry name must be text or null")
    products = {k: profile.get("products", {}).get(k, profile.get("products", {}).get(raw)) for k, raw in PRODUCTS.items()}
    if any(value is not None and not isinstance(value, bool) for value in products.values()):
        raise KnowledgeError("Product flags must be true, false or null")
    fts = profile.get("fts", {})
    fts_flag = fts.get("flag", profile.get("fts_flag"))
    countries = fts.get("countries", profile.get("fts_countries", []))
    if fts_flag is not None and not isinstance(fts_flag, bool):
        raise KnowledgeError("FTS flag must be true, false or null")
    if not isinstance(countries, list) or any(not isinstance(c, str) for c in countries):
        raise KnowledgeError("FTS countries must be a list of strings")
    region = profile.get("registration_region", profile.get("region_registration"))
    bindings = {"industry": [], "region": [], "fts": [], **{k: [] for k in PRODUCTS}}
    features, by_name = {}, {}
    for feature in row.get("features", []):
        if not feature.get("id") or feature["id"] in features:
            raise KnowledgeError("Company feature IDs must be present and unique")
        if feature.get("company_id", company_id) != company_id:
            raise KnowledgeError("Feature belongs to another company")
        version = feature.get("version", feature.get("profile_version"))
        payment_bound = (feature.get("name") in {"payment_incoming", "payment_outgoing"} and
                         feature.get("bound_profile_version") == row["profile_version"] and
                         version == feature.get("source_context_version") == row.get("payment_context_version") and bool(version))
        if version != reference_version and not payment_bound:
            raise KnowledgeError("Feature belongs to a different profile/context version")
        features[feature["id"]] = copy.deepcopy(feature)
        by_name.setdefault(feature["name"], []).append(feature)

    def bind(names, target, expected, transform=lambda x: x):
        for field in names:
            for feature in by_name.get(field, []):
                if transform(feature["value"]) != transform(expected):
                    raise KnowledgeError(f"Feature {field} disagrees with the company profile")
                bindings[target].append(feature["id"])

    bind(["industry"], "industry", {"code": code, "name": name})
    bind(["okved_code"], "industry", code)
    bind(["okved_name"], "industry", name)
    bind(["registration_region", "region_registration"], "region", region)
    bind(["fts"], "fts", {"flag": fts_flag, "countries": countries})
    bind(["fts_countries"], "fts", countries, lambda value: sorted(country(c) for c in value))
    for product, column in PRODUCTS.items():
        bind([column], product, products[product])
        for feature in by_name.get("products", []):
            if feature["value"].get(product) != products[product]:
                raise KnowledgeError("Product feature disagrees with profile")
            bindings[product].append(feature["id"])
    payment_sectors = []
    for direction in ("incoming", "outgoing"):
        for feature in by_name.get(f"payment_{direction}", []):
            if feature.get("method") != "nonself_all_flow_denominator_normalized_duplicates_once":
                continue
            value = feature["value"]
            coverage = value.get("known_profile_amount_share")
            if coverage is not None and (not isinstance(coverage, (int, float)) or not math.isfinite(coverage) or not 0 <= coverage <= 1):
                raise KnowledgeError("Invalid payment profile coverage")
            for sector in value.get("known_sectors", []):
                share = sector.get("share_of_all_flow")
                if share is None or not isinstance(share, (int, float)) or not math.isfinite(share) or not 0 < share <= 1:
                    continue
                payment_sectors.append({"code": sector.get("code"), "name": sector.get("name"), "share": share,
                                        "coverage": coverage, "direction": direction, "feature_id": feature["id"],
                                        "limitations": value.get("limitations", [])})
    limitations = list(row.get("limitations", []))
    if not features:
        limitations.append("company_feature_evidence_missing")
    if fts_flag is not True:
        limitations.append("fts_presence_not_confirmed")
    if fts_flag is False and countries:
        limitations.append("fts_country_without_flag")
    return {"id": company_id, "profile_version": row["profile_version"], "context_version": reference_version,
            "industry": {"code": code, "name": name}, "region": region, "products": products,
            "fts_flag": fts_flag, "countries": sorted(set(country(c) for c in countries)),
            "features": features, "bindings": bindings, "payment_sectors": payment_sectors,
            "quality_flags": profile.get("quality_flags", []), "limitations": sorted(set(limitations)),
            "role": profile.get("role"), "input_hash": digest(row)}
