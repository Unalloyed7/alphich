"""Strict provider output and public request contracts."""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


Topic = Literal["policy_rate", "inflation", "fuel", "fx", "trade_rules", "logistics", "regulation",
                "infrastructure", "labor", "consumer_demand", "sector", "commodities", "politics", "sports", "culture"]


class Attributes(Strict):
    topics: list[Topic] = Field(max_length=5)
    industry_codes: list[str] = Field(default_factory=list, max_length=5)
    countries: list[str] = Field(default_factory=list, max_length=5)
    regions: list[str] = Field(default_factory=list, max_length=5)
    geography: str | None = Field(default=None, max_length=12)
    kind: Literal["report", "rate_decision", "forecast", "regulatory_change"] = "report"
    forecast_horizon: str | None = None

    @field_validator("industry_codes")
    @classmethod
    def codes(cls, values):
        import re
        if any(not re.fullmatch(r"\d{2}(?:\.\d{1,2}){0,3}", v) for v in values):
            raise ValueError("Use OKVED prefixes, not invented taxonomy codes")
        return values

    @field_validator("forecast_horizon")
    @classmethod
    def horizon(cls, value):
        if value is not None and date.fromisoformat(value).isoformat() != value:
            raise ValueError("Use an ISO date")
        return value


class FactCandidate(Strict):
    quote: str = Field(min_length=6, max_length=1600)
    attributes: Attributes


class ExtractionDraft(Strict):
    facts: list[FactCandidate] = Field(max_length=10)


class ExtractionReview(Strict):
    accepted_indexes: list[int] = Field(max_length=10)
    issues: list[str] = Field(default_factory=list, max_length=12)


class Claim(Strict):
    id: str = Field(pattern=r"^c[1-9][0-9]?$", max_length=3)
    kind: Literal["news_fact", "company_fact", "bank_view", "model_inference"]
    text: str = Field(min_length=6, max_length=1000)
    evidence_ids: list[str] = Field(min_length=1, max_length=8)
    conditions: list[str] = Field(default_factory=list, max_length=5)


class NextStep(Strict):
    text: str = Field(min_length=6, max_length=500)
    basis_claim_ids: list[str] = Field(min_length=1, max_length=5)
    conditions: list[str] = Field(min_length=1, max_length=4)


class CardDraft(Strict):
    claims: list[Claim] = Field(min_length=1, max_length=12)
    risks: list[str] = Field(default_factory=list, max_length=5)
    opportunities: list[str] = Field(default_factory=list, max_length=5)
    next_steps: list[NextStep] = Field(default_factory=list, max_length=4)
    missing_data: list[str] = Field(default_factory=list, max_length=8)


class SearchBank(Strict):
    action: Literal["search_bank_materials"]
    query: str = Field(min_length=3, max_length=600)
    metric: str | None = Field(default=None, max_length=80)
    horizon: str | None = Field(default=None, max_length=10)


class RelatedEvents(Strict):
    action: Literal["search_related_events"]
    query: str = Field(min_length=3, max_length=300)


class GetFeatures(Strict):
    action: Literal["get_company_features", "get_payment_neighbors", "get_evidence"]
    ids: list[str] = Field(default_factory=list, max_length=5)


class Finish(Strict):
    action: Literal["finish"]
    draft: CardDraft


Action = Annotated[SearchBank | RelatedEvents | GetFeatures | Finish, Field(discriminator="action")]


class Decision(Strict):
    decision: Action


class FinalDecision(Strict):
    decision: Finish


class SemanticReview(Strict):
    supported_claim_ids: list[str] = Field(max_length=12)
    supported_action_indexes: list[int] = Field(max_length=4)
    issues: list[str] = Field(default_factory=list, max_length=12)


class InsightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    event_id: str = Field(min_length=1, max_length=40)
    as_of: datetime | None = None

    @field_validator("as_of")
    @classmethod
    def aware(cls, value):
        if value is not None:
            if value.utcoffset() is None:
                raise ValueError("as_of must include a timezone")
            value = value.astimezone(timezone.utc)
            if value > datetime.now(timezone.utc):
                raise ValueError("as_of cannot be in the future")
        return value


class QuestionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    question: str = Field(min_length=6, max_length=1000)
