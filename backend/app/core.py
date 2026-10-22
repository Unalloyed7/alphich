"""Persistent company profile store and reproducible CSV ingestion.

Organizer field meanings are bound to the supplied webinar clarification.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from sqlalchemy import JSON, ForeignKey, Integer, String, create_engine, delete, select
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from alphich.organizer_semantics import VERSION as SEMANTICS_VERSION, SOURCE_REF, revenue_metadata


class Base(DeclarativeBase):
    pass


class Company(Base):
    __tablename__ = "companies"
    id: Mapped[str] = mapped_column(String(100), primary_key=True)
    profile_version: Mapped[str] = mapped_column(String(64), nullable=False)
    source_rows: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    group_id: Mapped[str | None] = mapped_column(String(100))
    okved_code: Mapped[str | None] = mapped_column(String(32))
    okved_name: Mapped[str | None] = mapped_column(String(512))
    region: Mapped[str | None] = mapped_column(String(512))
    city: Mapped[str | None] = mapped_column(String(512))
    revenue_raw: Mapped[str | None] = mapped_column(String(128))
    client_type: Mapped[str | None] = mapped_column(String(100))
    products: Mapped[dict[str, bool | None]] = mapped_column(JSON, nullable=False)
    fts_flag: Mapped[bool | None] = mapped_column(JSON)
    countries: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    quality_flags: Mapped[list[str]] = mapped_column(JSON, nullable=False)


class RawCompanyRow(Base):
    __tablename__ = "company_raw_rows"
    row_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    company_id: Mapped[str | None] = mapped_column(String(100), index=True)
    content: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)


class QuarantinedCompany(Base):
    __tablename__ = "company_quarantine"
    company_id: Mapped[str] = mapped_column(String(100), primary_key=True)
    source_rows: Mapped[list[int]] = mapped_column(JSON, nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)


class DemoSession(Base):
    __tablename__ = "demo_sessions"
    token_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), ForeignKey("companies.id"), nullable=False)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    expires_at: Mapped[int] = mapped_column(Integer, nullable=False)


def database_url() -> str:
    return os.getenv("DATABASE_URL", "sqlite:///./alphich.db")


def make_session_factory(url: str | None = None):
    engine = create_engine(url or database_url(), pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        from sqlalchemy import event
        @event.listens_for(engine, "connect")
        def enable_fk(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return engine, sessionmaker(engine, expire_on_commit=False)


PRODUCT_COLUMNS = (
    "CL_PRODUCT_RKO_MARK", "CL_PRODUCT_GAR_MARK", "CL_PRODUCT_LOA_MARK",
    "CL_PRODUCT_LEA_MARK", "CL_PRODUCT_DEP_MARK", "CL_PRODUCT_ZPP_MARK",
    "FACTORING_MARK",
)
REQUIRED_COLUMNS = {"SYNTHETIC_U", "MAINOKVED_CODE", "REVENUE_COMPANY"}


def clean(value: str | None) -> str | None:
    val = (value or "").strip()
    return val or None


def flag(value: str | None) -> bool | None:
    val = clean(value)
    if val is None:
        return None
    if val in ("1", "1.0", "true", "True"):
        return True
    if val in ("0", "0.0", "false", "False"):
        return False
    return None


def normalize(row: dict[str, str | None]) -> dict[str, str | None]:
    return {k.strip(): clean(v) for k, v in row.items() if k is not None}


def profile_payload(company_id: str, row: dict[str, str | None], row_numbers: list[int]) -> dict:
    country_raw = row.get("FTS_COUNTRY_NAME")
    # Values are source fields, not verified import/export directions.
    countries = sorted({c.strip() for c in (country_raw or "").replace("|", ",").split(",") if c.strip()})
    fts = flag(row.get("FTS_FLAG"))
    issues: list[str] = []
    revenue = revenue_metadata(row.get("REVENUE_COMPANY"))
    issues.extend(revenue["quality_flags"])
    if fts is False and countries:
        issues.append("fts_country_without_flag")
    if fts is True and not countries:
        issues.append("fts_flag_without_country")
    for key in (*PRODUCT_COLUMNS, "FTS_FLAG"):
        if row.get(key) is not None and flag(row[key]) is None:
            issues.append(f"unrecognized_boolean:{key}")
    canonical = json.dumps({"row": row, "source_rows": row_numbers, "semantics": SEMANTICS_VERSION}, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return dict(
        id=company_id, profile_version=hashlib.sha256(canonical).hexdigest(),
        source_rows=row_numbers, group_id=row.get("GROUP_ID"),
        okved_code=row.get("MAINOKVED_CODE"), okved_name=row.get("MAINOKVED_NAME"),
        region=row.get("REGION"), city=row.get("CITY"),
        revenue_raw=revenue["amount"], client_type=row.get("CL_TYPE"),
        products={k: flag(row.get(k)) for k in PRODUCT_COLUMNS},
        fts_flag=fts, countries=countries, quality_flags=issues,
    )


def import_companies(csv_file: str | Path, session_factory, *, replace: bool = False) -> dict:
    """Import every source line, quarantine ALL contradictory versions of an ID.

    Atomic replacement keeps raw provenance, and refuses unexpected re-imports.
    Intended for the organizers' 85k-row company CSV, not payments/people.
    """
    path = Path(csv_file)
    first: dict[str, tuple[dict[str, str | None], list[int]]] = {}
    conflicts: dict[str, list[int]] = {}
    raw: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if not REQUIRED_COLUMNS.issubset(set(reader.fieldnames or [])):
            raise ValueError(f"Missing required columns: {sorted(REQUIRED_COLUMNS - set(reader.fieldnames or []))}")
        for row_number, source in enumerate(reader, start=2):
            row = normalize(source)
            uid = row.get("SYNTHETIC_U")
            raw.append({"row_number": row_number, "company_id": uid, "content": row})
            if not uid:
                continue
            if uid not in first:
                first[uid] = (row, [row_number])
            else:
                previous, positions = first[uid]
                positions.append(row_number)
                if row != previous:
                    conflicts[uid] = positions
    if not raw:
        raise ValueError("Empty CSV is not a valid company snapshot")
    good = [profile_payload(uid, row, positions) for uid, (row, positions) in first.items() if uid not in conflicts]
    bad = [dict(company_id=uid, source_rows=first[uid][1], reason="conflicting_profiles") for uid in conflicts]
    with session_factory.begin() as db:
        if db.scalar(select(Company.id).limit(1)) is not None and not replace:
            raise ValueError("Companies already imported; pass --replace for an atomic refresh")
        # Clear sessions first: an imported profile's version may have changed.
        db.execute(delete(DemoSession))
        db.execute(delete(QuarantinedCompany))
        db.execute(delete(Company))
        db.execute(delete(RawCompanyRow))
        for start in range(0, len(raw), 1000):
            db.bulk_insert_mappings(RawCompanyRow, raw[start:start + 1000])
        for start in range(0, len(good), 1000):
            db.bulk_insert_mappings(Company, good[start:start + 1000])
        if bad:
            db.bulk_insert_mappings(QuarantinedCompany, bad)
    return {"raw_rows": len(raw), "unique_ids": len(first), "companies": len(good),
            "quarantined_ids": len(bad), "missing_id_rows": sum(not r["company_id"] for r in raw)}


def feature_evidence(company: Company) -> list[dict]:
    """Version-bound, reviewable facts; no unsupported financial inference."""
    revenue = revenue_metadata(company.revenue_raw)
    values: list[tuple[str, object]] = [
        ("okved_code", company.okved_code), ("okved_name", company.okved_name),
        ("region_registration", company.region), ("revenue_raw", revenue["amount"]),
        ("group_id", company.group_id),
        ("fts_countries", company.countries),
    ] + [(key, value) for key, value in company.products.items()]
    if revenue["amount"] is not None:
        values.extend(("revenue_" + key, revenue[key]) for key in ("unit", "currency", "period", "scope"))
    out = []
    for name, value in values:
        if value is None or value == []:
            continue
        feature_id = "feature_" + hashlib.sha256(f"{company.id}:{company.profile_version}:{SEMANTICS_VERSION}:{name}".encode()).hexdigest()[:24]
        out.append({"id": feature_id, "name": name, "value": value,
                    "unit": "RUB" if name == "revenue_raw" else None, "method": "source_field_with_organizer_clarification",
                    "source_refs": [*[f"df_Uid_data_hack.csv:row:{r}" for r in company.source_rows], SOURCE_REF],
                    "profile_version": company.profile_version})
    return out
