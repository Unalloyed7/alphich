"""Prepare auditable company contexts without loading payment CSV into a list.

Money is kept as Decimal internally and serialized as decimal strings. The
organizer webinar establishes revenue in RUB for 2025 and YTD payments in RUB.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterator
from .organizer_semantics import VERSION as SEMANTICS_VERSION, SOURCE_REF, PAYMENT_PERIOD, revenue_metadata

SCHEMA_VERSION = "company-context-v1"
METHOD_VERSION = "company-features-v2-webinar"
PROFILE_FILE = "df_Uid_data_hack.csv"
PAYMENT_FILE = "df_Uid_payment_hack.csv"
LINK_FILE = "df_Fid_Uid_link_hack.csv"
PRODUCT_COLUMNS = {
    "rko": "CL_PRODUCT_RKO_MARK",
    "guarantee": "CL_PRODUCT_GAR_MARK",
    "loan": "CL_PRODUCT_LOA_MARK",
    "lease": "CL_PRODUCT_LEA_MARK",
    "deposit": "CL_PRODUCT_DEP_MARK",
    "payroll": "CL_PRODUCT_ZPP_MARK",
    "factoring": "FACTORING_MARK",
}
PROFILE_COLUMNS = {
    "SYNTHETIC_U", "MAINOKVED_CODE", "MAINOKVED_NAME", "REGION", "CITY",
    "CL_TYPE", "REVENUE_COMPANY", "GROUP_ID", "FTS_FLAG", "FTS_COUNTRY_NAME",
    "AMOUNT_WORKERS_FNS_GRADATION", *PRODUCT_COLUMNS.values(),
}


class DataError(ValueError):
    """An input or artifact cannot safely be interpreted."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def decimal_value(text: str, name: str) -> Decimal:
    try:
        value = Decimal(text)
    except InvalidOperation as error:
        raise DataError(f"{name}: expected a decimal number") from error
    if not value.is_finite() or value < 0:
        raise DataError(f"{name}: expected a finite nonnegative number")
    return value


def flag(text: str, name: str, issues: list[str]) -> bool | None:
    if not text:
        issues.append(f"missing:{name}")
        return None
    if text in {"0", "0.0"}:
        return False
    if text in {"1", "1.0"}:
        return True
    issues.append(f"invalid_flag:{name}")
    return None


def csv_rows(path: Path, required: set[str]) -> Iterator[tuple[int, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter=";")
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise DataError(f"{path.name}: duplicate column names")
        missing = required - set(fields)
        if missing:
            raise DataError(f"{path.name}: missing columns: {', '.join(sorted(missing))}")
        for record, row in enumerate(reader, start=1):
            if None in row or any(value is None for value in row.values()):
                raise DataError(f"{path.name}, record {record}: incorrect number of columns")
            yield record, {key: value.strip() for key, value in row.items()}


@dataclass(slots=True)
class Company:
    id: str
    source_record: int
    signature: str
    profile: dict[str, Any]
    revenue: Decimal | None


def parse_company(record: int, row: dict[str, str]) -> Company:
    company_id = row["SYNTHETIC_U"]
    if not company_id:
        raise DataError("missing company ID")
    issues: list[str] = []
    products = {name: flag(row[column], column, issues) for name, column in PRODUCT_COLUMNS.items()}
    fts = flag(row["FTS_FLAG"], "FTS_FLAG", issues)
    countries = sorted({country.strip() for country in row["FTS_COUNTRY_NAME"].split(",") if country.strip()})
    if fts is True and not countries:
        issues.append("fts_flag_without_country")
    if fts is False and countries:
        issues.append("fts_country_without_flag")
    revenue_info = revenue_metadata(row["REVENUE_COMPANY"])
    issues.extend(revenue_info["quality_flags"])
    revenue = Decimal(revenue_info["amount"]) if revenue_info["amount"] is not None else None
    if revenue is None:
        issues.append("missing:REVENUE_COMPANY")
    for name in ("MAINOKVED_CODE", "MAINOKVED_NAME", "REGION", "GROUP_ID"):
        if not row[name]:
            issues.append(f"missing:{name}")
    profile = {
        "id": company_id,
        "client_type": row["CL_TYPE"] or None,
        "industry": {"code": row["MAINOKVED_CODE"] or None, "name": row["MAINOKVED_NAME"] or None},
        "registration_region": row["REGION"] or None,
        "registration_city": row["CITY"] or None,
        "group_id": row["GROUP_ID"] or None,
        "revenue_raw": decimal_text(revenue) if revenue is not None else None,
        "revenue_unit": "RUB", "revenue_currency": "RUB", "revenue_period": "2025",
        "revenue_scope": revenue_info["scope"], "semantics_source_ref": SOURCE_REF,
        "group_semantics": "ownership_or_management_link_not_shared_debt_or_operations",
        "snapshot_at": None,
        "employees_fns_band": row["AMOUNT_WORKERS_FNS_GRADATION"] or None,
        "products": products,
        "fts": {"flag": fts, "countries": countries},
        "quality_flags": issues,
        "source_record": record,
        # Preserve unmapped columns without assigning undocumented semantics.
        "source_fields": row,
    }
    return Company(company_id, record, stable_hash(row), profile, revenue)


def load_companies(path: Path) -> tuple[dict[str, Company], dict[str, Any], list[dict[str, Any]]]:
    companies: dict[str, Company] = {}
    first_signature: dict[str, str] = {}
    records_by_id: dict[str, list[int]] = defaultdict(list)
    conflicting: set[str] = set()
    quarantine: list[dict[str, Any]] = []
    issues: Counter[str] = Counter()
    rows = 0
    for record, row in csv_rows(path, PROFILE_COLUMNS):
        rows += 1
        company_id = row["SYNTHETIC_U"]
        if company_id:
            records_by_id[company_id].append(record)
            signature = stable_hash(row)
            if company_id in first_signature and first_signature[company_id] != signature:
                conflicting.add(company_id)
            first_signature.setdefault(company_id, signature)
        try:
            company = parse_company(record, row)
        except DataError as error:
            quarantine.append({"dataset": path.name, "record": record, "company_id": company_id or None, "reason": str(error)})
            continue
        issues.update(company.profile["quality_flags"])
        companies.setdefault(company.id, company)
    for company_id in sorted(conflicting):
        companies.pop(company_id, None)
        quarantine.append({"dataset": path.name, "company_id": company_id, "records": records_by_id[company_id], "reason": "conflicting_company_profiles"})
    report = {
        "rows": rows,
        "unique_ids": len(records_by_id),
        "usable_profiles": len(companies),
        "conflicting_company_ids": sorted(conflicting),
        "additional_id_rows": sum(len(records) - 1 for records in records_by_id.values()),
        "quality_flag_counts_raw_rows": dict(sorted(issues.items())),
        "client_type_counts_usable": dict(sorted(Counter(company.profile["client_type"] or "UNKNOWN" for company in companies.values()).items())),
    }
    return companies, report, quarantine


@dataclass(slots=True)
class Edge:
    amount: Decimal = Decimal(0)
    count: int = 0
    records: int = 0


@dataclass(slots=True)
class Flow:
    edges: dict[str, Edge] = field(default_factory=dict)
    exact_duplicate_rows: int = 0
    repeated_pair_rows: int = 0

    def add(self, neighbor: str, amount: Decimal, count: int) -> None:
        edge = self.edges.setdefault(neighbor, Edge())
        edge.amount += amount
        edge.count += count
        edge.records += 1


def ratio(numerator: Decimal, denominator: Decimal) -> float | None:
    return round(float(numerator / denominator), 10) if denominator else None


def summarize_flow(flow: Flow, owner: Company, companies: dict[str, Company], top_n: int) -> dict[str, Any]:
    amount = sum((edge.amount for edge in flow.edges.values()), Decimal(0))
    known_amount = Decimal(0)
    within_group = Decimal(0)
    unknown_count = 0
    sectors: dict[tuple[str, str], Decimal] = defaultdict(Decimal)
    neighbors = []
    for neighbor, edge in flow.edges.items():
        profile = companies.get(neighbor)
        if profile:
            known_amount += edge.amount
            code = profile.profile["industry"]["code"]
            name = profile.profile["industry"]["name"]
            if code and name:
                sectors[(code, name)] += edge.amount
            if owner.profile["group_id"] and owner.profile["group_id"] == profile.profile["group_id"]:
                within_group += edge.amount
        else:
            unknown_count += 1
        neighbors.append({"company_id": neighbor, "amount_raw": decimal_text(edge.amount), "payment_count": edge.count,
                          "share_of_all_flow": ratio(edge.amount, amount), "profile_known": profile is not None,
                          "industry": profile.profile["industry"] if profile else None})
    neighbors.sort(key=lambda item: (-Decimal(item["amount_raw"]), item["company_id"]))
    sector_rows = [
        {"code": code, "name": name, "amount_raw": decimal_text(value), "share_of_all_flow": ratio(value, amount)}
        for (code, name), value in sorted(sectors.items(), key=lambda item: (-item[1], item[0]))
    ]
    limitations = ["payment_period_end_and_year_unknown", "payment_purpose_unknown"]
    if not flow.edges:
        limitations.append("no_observed_payments")
    elif not amount:
        limitations.append("zero_total_payment_amount")
    if flow.exact_duplicate_rows or flow.repeated_pair_rows:
        limitations.append("repeated_payment_rows_demo_policy")
    return {
        "amount_raw": decimal_text(amount), "currency": "RUB", "period": dict(PAYMENT_PERIOD),
        "semantics_source_ref": SOURCE_REF,
        "payment_count": sum(edge.count for edge in flow.edges.values()),
        "counterparty_count": len(flow.edges), "unknown_profile_counterparty_count": unknown_count,
        "known_profile_amount_share": ratio(known_amount, amount),
        "within_known_group_amount_share": ratio(within_group, amount),
        "top_counterparty_share": neighbors[0]["share_of_all_flow"] if neighbors else None,
        "top_counterparties": neighbors[:top_n], "known_sectors": sector_rows[:top_n],
        "top_n": top_n, "shares_denominator_raw": decimal_text(amount),
        "normalized_duplicate_rows_skipped": flow.exact_duplicate_rows,
        "additional_pair_rows_aggregated": flow.repeated_pair_rows,
        "limitations": limitations,
    }


def load_payments(path: Path, companies: dict[str, Company], selected: set[str]) -> tuple[dict[str, dict[str, Flow]], dict[str, Any], list[dict[str, Any]]]:
    flows = {company_id: {"incoming": Flow(), "outgoing": Flow()} for company_id in selected}
    seen_pairs: set[str] = set()
    seen_rows: set[str] = set()
    nodes: set[str] = set()
    stats: Counter[str] = Counter({key: 0 for key in (
        "rows", "valid_rows", "invalid_rows", "self_payment_rows",
        "additional_directed_pair_rows", "normalized_duplicate_rows_skipped",
    )})
    quarantine = []
    raw_amount, self_amount = Decimal(0), Decimal(0)
    for record, row in csv_rows(path, {"PAYER", "BENEFICIARY", "PAY_AMT", "PAY_CNT"}):
        stats["rows"] += 1
        payer, beneficiary = row["PAYER"], row["BENEFICIARY"]
        try:
            if not payer or not beneficiary:
                raise DataError("missing payment party ID")
            amount = decimal_value(row["PAY_AMT"], "PAY_AMT")
            count_value = decimal_value(row["PAY_CNT"], "PAY_CNT")
            if count_value != count_value.to_integral_value():
                raise DataError("PAY_CNT: expected an integer")
            count = int(count_value)
        except DataError as error:
            stats["invalid_rows"] += 1
            quarantine.append({"dataset": path.name, "record": record, "reason": str(error)})
            continue
        stats["valid_rows"] += 1
        nodes.update((payer, beneficiary))
        raw_amount += amount
        if payer == beneficiary:
            stats["self_payment_rows"] += 1
            self_amount += amount
        pair = stable_hash([payer, beneficiary])
        signature = stable_hash([payer, beneficiary, decimal_text(amount), count])
        repeated = pair in seen_pairs
        duplicate = signature in seen_rows
        stats["additional_directed_pair_rows"] += int(repeated)
        stats["normalized_duplicate_rows_skipped"] += int(duplicate)
        seen_pairs.add(pair)
        seen_rows.add(signature)
        if payer == beneficiary:
            continue
        targets = []
        if payer in selected:
            targets.append((flows[payer]["outgoing"], beneficiary))
        if beneficiary in selected:
            targets.append((flows[beneficiary]["incoming"], payer))
        for flow, neighbor in targets:
            if duplicate:
                flow.exact_duplicate_rows += 1
                continue
            flow.repeated_pair_rows += int(repeated)
            flow.add(neighbor, amount, count)
    report = {**stats, "nodes": len(nodes), "nodes_with_usable_profile": len(nodes & companies.keys()),
              "raw_valid_amount": decimal_text(raw_amount), "raw_self_amount": decimal_text(self_amount),
              "raw_self_amount_share": ratio(self_amount, raw_amount),
              "duplicate_policy": "identical_normalized_rows_once_other_repeated_pairs_sum_with_warning",
              "currency": "RUB", "period": dict(PAYMENT_PERIOD), "semantics_source_ref": SOURCE_REF,
              "fingerprint": "sha256_normalized_values"}
    return flows, report, quarantine


def load_links(path: Path, selected: set[str]) -> tuple[dict[str, dict[str, int]], dict[str, int], list[dict[str, Any]]]:
    people: dict[str, set[str]] = defaultdict(set)
    selected_people: dict[str, set[str]] = defaultdict(set)
    stats: Counter[str] = Counter({"rows": 0, "invalid_rows": 0, "duplicate_links_skipped": 0})
    quarantine = []
    for record, row in csv_rows(path, {"SYNTHETIC_F", "SYNTHETIC_U"}):
        stats["rows"] += 1
        person, company_id = row["SYNTHETIC_F"], row["SYNTHETIC_U"]
        if not person or not company_id:
            stats["invalid_rows"] += 1
            quarantine.append({"dataset": path.name, "record": record, "reason": "missing_link_party_id"})
            continue
        if company_id in people[person]:
            stats["duplicate_links_skipped"] += 1
        people[person].add(company_id)
        if company_id in selected:
            selected_people[company_id].add(person)
    result = {}
    for company_id in selected:
        related: set[str] = set()
        for person in selected_people[company_id]:
            related.update(people[person])
        related.discard(company_id)
        result[company_id] = {"linked_person_count": len(selected_people[company_id]), "other_linked_company_count": len(related)}
    return result, {**stats, "unique_people": len(people)}, quarantine


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: Iterator[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as output:
        for value in values:
            output.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def prepare_data(input_dir: Path, output_dir: Path, company_ids: list[str] | None = None,
                 min_revenue_raw: Decimal = Decimal("15000000000"), top_n: int = 10) -> dict[str, Any]:
    if top_n < 1:
        raise DataError("top_n must be positive")
    if not min_revenue_raw.is_finite() or min_revenue_raw < 0:
        raise DataError("min_revenue_raw must be finite and nonnegative")
    if output_dir.exists():
        raise DataError(f"Output already exists: {output_dir}. Choose a new directory to preserve the previous snapshot.")
    paths = {name: input_dir / name for name in (PROFILE_FILE, PAYMENT_FILE, LINK_FILE)}
    hashes = {name: sha256_file(path) for name, path in paths.items()}
    dataset_version = stable_hash({"inputs": hashes, "organizer_semantics": SEMANTICS_VERSION})
    companies, profile_report, quarantine = load_companies(paths[PROFILE_FILE])
    if company_ids:
        selected = set(company_ids)
        unknown = selected - companies.keys()
        if unknown:
            raise DataError(f"Unknown or quarantined companies: {', '.join(sorted(unknown))}")
    else:
        selected = {company.id for company in companies.values()
                    if company.profile["client_type"] == "CLIENT" and company.revenue is not None and company.revenue > min_revenue_raw}
    if not selected:
        raise DataError("No companies match the selection; use --company-id or another --min-revenue-raw")
    flows, payment_report, payment_quarantine = load_payments(paths[PAYMENT_FILE], companies, selected)
    links, link_report, link_quarantine = load_links(paths[LINK_FILE], selected)
    quarantine.extend(payment_quarantine)
    quarantine.extend(link_quarantine)
    group_sizes = Counter(company.profile["group_id"] for company in companies.values() if company.profile["group_id"])
    config = {
        "company_ids": sorted(selected), "top_n": top_n, "method_version": METHOD_VERSION,
        "mode": "explicit_ids" if company_ids else "client_raw_revenue",
        "client_type": None if company_ids else "CLIENT",
        "min_revenue_raw": None if company_ids else decimal_text(min_revenue_raw),
    }
    context_version = stable_hash({"dataset": dataset_version, "config": config})

    def feature(company_id: str, name: str, value: Any, datasets: list[str], method: str) -> dict[str, Any]:
        return {"id": f"feature:{company_id}:{name}:{context_version[:16]}", "company_id": company_id,
                "name": name, "value": value, "version": context_version, "method": method,
                "source_refs": [{"dataset": name, "sha256": hashes[name]} for name in datasets]}

    def context(company_id: str) -> dict[str, Any]:
        company = companies[company_id]
        profile = company.profile
        features = [feature(company_id, "industry", profile["industry"], [PROFILE_FILE], "profile_main_okved"),
                    feature(company_id, "registration_region", profile["registration_region"], [PROFILE_FILE], "profile_registration_region"),
                    feature(company_id, "products", profile["products"], [PROFILE_FILE], "profile_product_flags_unknown_preserved"),
                    feature(company_id, "fts", profile["fts"], [PROFILE_FILE], "profile_fts_with_quality_flags"),
                    feature(company_id, "group_profile_count", group_sizes[profile["group_id"]] if profile["group_id"] else None,
                            [PROFILE_FILE], "usable_profiles_in_same_declared_group"),
                    feature(company_id, "person_links", links[company_id], [LINK_FILE], "shared_person_no_role_inferred")]
        for name in ("revenue_raw", "revenue_unit", "revenue_currency", "revenue_period", "revenue_scope", "group_id"):
            if profile.get(name) is not None:
                item = feature(company_id, name, profile[name], [PROFILE_FILE], "source_field_with_organizer_clarification")
                item["semantics_source_ref"] = SOURCE_REF
                features.append(item)
        for direction in ("incoming", "outgoing"):
            value = summarize_flow(flows[company_id][direction], company, companies, top_n)
            features.append(feature(company_id, f"payment_{direction}", value, [PAYMENT_FILE, PROFILE_FILE],
                                    "nonself_all_flow_denominator_normalized_duplicates_once"))
        return {"schema_version": SCHEMA_VERSION, "company_id": company_id, "profile_version": dataset_version,
                "context_version": context_version, "profile": profile, "features": features,
                "limitations": ["company_snapshot_date_unknown", "revenue_company_or_group_scope_unknown", "market_share_unavailable",
                                "credit_terms_unknown", "person_link_role_unknown"]}

    report = {"schema_version": SCHEMA_VERSION, "dataset_version": dataset_version,
              "profiles": profile_report, "payments": payment_report, "person_links": link_report,
              "selected_company_ids": sorted(selected), "quarantined_entries": len(quarantine)}
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".alphich-data-", dir=output_dir.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        write_jsonl(staging / "companies.jsonl", (companies[key].profile for key in sorted(companies)))
        write_jsonl(staging / "contexts.jsonl", (context(key) for key in sorted(selected)))
        write_jsonl(staging / "quarantine.jsonl", iter(quarantine))
        write_json(staging / "quality.json", report)
        manifest = {"schema_version": SCHEMA_VERSION, "method_version": METHOD_VERSION,
                    "dataset_version": dataset_version, "context_version": context_version,
                    "inputs": [{"name": name, "sha256": digest} for name, digest in sorted(hashes.items())],
                    "selection": config,
                    "files": {path.name: sha256_file(path) for path in sorted(staging.iterdir())},
                    "money_encoding": "decimal_string", "money_currency": "RUB", "payment_period": dict(PAYMENT_PERIOD),
                    "organizer_semantics": SEMANTICS_VERSION, "semantics_source_ref": SOURCE_REF}
        write_json(staging / "manifest.json", manifest)
        os.rename(staging, output_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare versioned company contexts for Alphich")
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing the three organizer CSV files")
    parser.add_argument("--output-dir", type=Path, required=True, help="New snapshot directory; existing data is never overwritten")
    parser.add_argument("--company-id", action="append", help="Include a particular usable company; repeat to include more")
    parser.add_argument("--min-revenue-raw", default="15000000000", help="Default CLIENT selection: raw revenue strictly greater than this value")
    parser.add_argument("--top-n", type=int, default=10, help="Number of neighbors/sectors to expose; denominators always use all flows")
    args = parser.parse_args(argv)
    try:
        minimum = decimal_value(args.min_revenue_raw, "min-revenue-raw")
        report = prepare_data(args.input_dir, args.output_dir, args.company_id, minimum, args.top_n)
    except (DataError, OSError, UnicodeError, csv.Error) as error:
        print(f"alphich-data: {error}", file=sys.stderr)
        return 2
    print(json.dumps({"output_dir": str(args.output_dir), "usable_profiles": report["profiles"]["usable_profiles"],
                      "selected_companies": len(report["selected_company_ids"]),
                      "payment_rows": report["payments"].get("rows", 0),
                      "quarantined_entries": report["quarantined_entries"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
