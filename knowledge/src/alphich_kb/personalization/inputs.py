"""Explicit visibility, immutable event snapshots, and source-bound annotations."""

import copy
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from ..common import KnowledgeError, digest


def instant(value):
    if not isinstance(value, str):
        raise KnowledgeError("A timezone-aware ISO timestamp is required")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise KnowledgeError("A timezone-aware ISO timestamp is required") from error
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise KnowledgeError("Timestamps must include a timezone")
    return parsed.astimezone(timezone.utc)


def timestamp(value):
    return instant(value).isoformat() if value else None


def words(value):
    return " ".join(str(value or "").casefold().replace("ё", "е").split())


def country(value):
    return {"кнр": "китай", "china": "китай", "российская федерация": "россия", "ru": "россия",
            "russia": "россия", "us": "сша", "usa": "сша"}.get(words(value), words(value))


def industry_matches(company_code, prefix):
    if not company_code or not prefix:
        return False
    if company_code == prefix or company_code.startswith(prefix + "."):
        return True
    # OKVED subgroup 49.4 includes 49.41: this level adds a digit without a dot.
    parent, dot, last = prefix.rpartition(".")
    return bool(dot and len(last) == 1 and company_code.startswith(parent + "." + last)
                and len(company_code.split(".")[len(parent.split("."))]) == 2)


@dataclass(frozen=True)
class SelectionScope:
    company_id: str
    as_of: str
    allowed_event_ids: frozenset[str]
    role: str = "cfo"
    market_country: str = "RU"

    def __post_init__(self):
        object.__setattr__(self, "as_of", instant(self.as_of).isoformat())
        if self.role not in {"cfo", "ceo"} or not self.company_id:
            raise KnowledgeError("A company ID and cfo/ceo role are required")
        if isinstance(self.allowed_event_ids, str) or any(not isinstance(i, str) for i in self.allowed_event_ids):
            raise KnowledgeError("Event access scope must be a collection of string IDs")
        object.__setattr__(self, "allowed_event_ids", frozenset(self.allowed_event_ids))

    def excludes(self, event):
        if event["id"] not in self.allowed_event_ids or event["company_id"] not in {None, self.company_id}:
            return "access_denied"
        if not event["available_at"]:
            return "availability_unknown"
        if instant(event["available_at"]) > instant(self.as_of):
            return "future_source_version"
        publications = [event["published_at"], *(s["published_at"] for s in event["sources"])]
        if any(value and instant(value) > instant(self.as_of) for value in publications):
            return "future_publication"
        return None


def normalize_event(row):
    event_id, version = row.get("id"), row.get("version")
    if not isinstance(event_id, str) or not event_id or version is None or not isinstance(row.get("title"), str) or not row["title"].strip():
        raise KnowledgeError("Each event needs id, version and title")
    owner = row.get("company_id")
    sources = []
    for raw in row.get("sources", []):
        if not raw.get("id") or not isinstance(raw.get("text"), str) or not raw["text"].strip():
            raise KnowledgeError("Supply source IDs and full text; backend event metadata alone is insufficient")
        if raw.get("company_id", owner) != owner:
            raise KnowledgeError("A source cannot belong to another company")
        if len(raw["text"]) > 100_000:
            raise KnowledgeError("A source exceeds the 100000-character publication limit")
        sources.append({"id": raw["id"], "version": str(raw.get("version", 1)), "text": raw["text"],
                        "url": raw.get("url"), "available_at": timestamp(raw.get("available_at") or raw.get("source_available_at")
                                                                 or raw.get("received_at") or row.get("available_at")),
                        "published_at": timestamp(raw.get("published_at")),
                        "completeness": raw.get("completeness", "unknown"), "synthetic": bool(raw.get("synthetic", False))})
    if not sources or len({s["id"] for s in sources}) != len(sources):
        raise KnowledgeError("An event needs distinct source records with text")
    by_source = {s["id"]: s for s in sources}
    facts, rejected = [], []
    for raw in row.get("facts", []):
        if not isinstance(raw, dict):
            rejected.append({"fact_id": None, "reason": "fact_must_be_an_object"})
            continue
        try:
            source = by_source[raw["source_id"]]
            span = raw["source_span"]
            start, end = span["start"], span["end"]
            if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(source["text"]):
                raise KnowledgeError("invalid_span")
            if raw["text"] != source["text"][start:end]:
                raise KnowledgeError("quote_mismatch")
            if str(raw.get("source_version", source["version"])) != source["version"]:
                raise KnowledgeError("source_version_mismatch")
            if raw.get("verification_status") not in {"verified", "human_verified"}:
                raise KnowledgeError("fact_unverified")
            available = timestamp(raw.get("source_available_at") or source["available_at"])
            if not available or not source["available_at"] or instant(available) < instant(source["available_at"]):
                raise KnowledgeError("fact_availability_invalid")
            attributes = copy.deepcopy(raw.get("attributes", {}))
            if not isinstance(attributes, dict):
                raise KnowledgeError("fact_attributes_must_be_an_object")
            for name in ("topics", "industry_codes", "countries", "regions"):
                values = attributes.get(name, [])
                if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
                    raise KnowledgeError("invalid_fact_attributes")
                attributes[name] = sorted(set(values))
            if any(not re.fullmatch(r"\d{2}(?:\.\d{1,2}){0,3}", code) for code in attributes["industry_codes"]):
                raise KnowledgeError("invalid_industry_code")
            horizon = attributes.get("forecast_horizon")
            if horizon is not None and (not isinstance(horizon, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", horizon)):
                raise KnowledgeError("forecast_horizon_must_be_an_iso_date")
            if horizon:
                datetime.strptime(horizon, "%Y-%m-%d")
            dimensions = {
                "geographic_scope": {"unknown", "national", "global_market", "regional", "local"},
                "market_scope": {"unknown", "single_case", "sector_wide"},
                "impact_scale": {"unknown", "limited", "material"},
                "product_scope": {"unknown", "broad_core", "niche"},
                "event_status": {"unknown", "reported", "proposal", "forecast", "in_force", "announcement"},
                "business_effect": {"unknown", "core_market", "operating_requirement", "financing"},
                "applicability": {"unknown", "whole_industry", "conditional"},
            }
            for name, allowed in dimensions.items():
                value = attributes.get(name, "unknown")
                if not isinstance(value, str) or value not in allowed:
                    raise KnowledgeError("invalid_business_scope_attribute")
                attributes[name] = value
            for name in ("effective_from", "reference_period_end"):
                period = attributes.get(name)
                if period is not None:
                    if not isinstance(period, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", period):
                        raise KnowledgeError(name + "_must_be_an_iso_date")
                    datetime.strptime(period, "%Y-%m-%d")
            if not raw.get("id"):
                raise KnowledgeError("fact_id_missing")
            facts.append({"id": raw["id"], "text": raw["text"], "source_id": source["id"],
                          "source_version": source["version"], "source_span": {"start": start, "end": end},
                          "source_available_at": available, "attributes": attributes,
                          "verification_status": raw["verification_status"]})
        except (KnowledgeError, KeyError, TypeError, ValueError) as error:
            rejected.append({"fact_id": raw.get("id"), "reason": str(error)})
    if len({f["id"] for f in facts}) != len(facts):
        raise KnowledgeError("Duplicate fact IDs within an event")
    visibility = [s["available_at"] for s in sources]
    if row.get("available_at"):
        visibility.append(timestamp(row["available_at"]))
    available_at = max(visibility, key=instant) if all(visibility) else None
    event = {"id": str(event_id), "version": str(version), "company_id": owner, "title": row["title"],
             "published_at": timestamp(row.get("published_at")), "available_at": available_at,
             "event_at": row.get("event_at"), "sources": sources, "facts": facts, "rejected_facts": rejected,
             "text": "\n\n".join(s["text"] for s in sources), "completeness": row.get("completeness", "unknown")}
    event["snapshot_hash"] = digest(event)
    return event


def from_backend_event(event, source_records):
    """Join GET /events/{id} with authorized GET /sources/{id} responses."""
    by_id = {s["id"]: s for s in source_records}
    sources = []
    for source_id in event["source_ids"]:
        if source_id not in by_id:
            raise KnowledgeError(f"Missing authorized source body: {source_id}")
        source = by_id[source_id]
        if source.get("company_id") != event.get("company_id"):
            raise KnowledgeError("Backend event and source have different owners")
        sources.append({**source, "available_at": source.get("received_at")})
    return {**event, "sources": sources}
