"""Explicit organizer gateway IDs and confirmed RUB prices; fail closed."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation, ROUND_CEILING
from urllib.parse import urlsplit

CASE_MODELS = {"gemma4-26b-a4b", "glm-5.3-flash", "qwen3.6-35b-a3b", "deepseek-v4.1-flash",
               "qwen3-vl-30b-a3b-instruct", "kimi-k2.6"}


class AnalysisError(Exception):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class SchemaFailure(AnalysisError):
    def __init__(self, issues):
        super().__init__("llm_schema_invalid")
        self.issues = issues


def micros(value):
    return int((Decimal(str(value)) * 1_000_000).to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class Model:
    id: str
    case_model: str
    input_rub_per_million: Decimal | None = None
    output_rub_per_million: Decimal | None = None
    input_limit: int = 32768
    output_limit: int = 2500
    reasoning_enabled: bool | None = None
    response_mode: str = "text"

    def cost(self, input_tokens, output_tokens):
        if self.input_rub_per_million is None or self.output_rub_per_million is None:
            raise AnalysisError("token_prices_unavailable")
        # A million tokens and a million micro-rubles cancel; always round up.
        return int((self.input_rub_per_million * input_tokens + self.output_rub_per_million * output_tokens)
                   .to_integral_value(rounding=ROUND_CEILING))


@dataclass(frozen=True)
class Settings:
    base_url: str = ""
    api_key: str = field(default="", repr=False)
    extraction_model: str = ""
    analysis_model: str = ""
    models: dict[str, Model] = field(default_factory=dict)
    prices_confirmed: bool = False
    budget_mode: str = "token_prices"
    receipt_rub_per_unit: Decimal | None = None
    daily_limit_micros: int = 500_000_000
    timeout_seconds: float = 45
    max_calls: int = 8
    max_tools: int = 8
    bank_index: str = "/data/processed/bank-index-v1"
    model_dir: str = "/data/models/e5-base"
    model_manifest: str = "/srv/knowledge/embedding-model.json"
    allowed_bank_ids: frozenset[str] | None = None

    @classmethod
    def from_env(cls):
        try:
            catalog = json.loads(os.getenv("LLM_MODEL_CATALOG_JSON", "[]"))
            models = {}
            for row in catalog:
                model = Model(row["id"], row["case_model"],
                              Decimal(str(row["input_rub_per_million"])) if row.get("input_rub_per_million") is not None else None,
                              Decimal(str(row["output_rub_per_million"])) if row.get("output_rub_per_million") is not None else None,
                              int(row.get("input_limit", 32768)),
                              int(row.get("output_limit", 2500)), row.get("reasoning_enabled"), row.get("response_mode", "text"))
                if model.id in models:
                    raise ValueError("duplicate_model")
                models[model.id] = model
            result = cls(base_url=os.getenv("LLM_BASE_URL", "").rstrip("/"), api_key=os.getenv("LLM_API_KEY", ""),
                         extraction_model=os.getenv("LLM_EXTRACTION_MODEL", ""), analysis_model=os.getenv("LLM_ANALYSIS_MODEL", ""),
                         models=models, prices_confirmed=os.getenv("LLM_PRICES_CONFIRMED") == "1",
                         budget_mode=os.getenv("LLM_BUDGET_MODE", "token_prices"),
                         receipt_rub_per_unit=Decimal(os.environ["LLM_RECEIPT_RUB_PER_UNIT"])
                         if os.getenv("LLM_RECEIPT_RUB_PER_UNIT") else None,
                         daily_limit_micros=micros(os.getenv("LLM_DAILY_LIMIT_RUB", "500")),
                         timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "45")),
                         max_calls=int(os.getenv("LLM_MAX_CALLS", "8")), max_tools=int(os.getenv("LLM_MAX_TOOLS", "8")),
                         bank_index=os.getenv("BANK_INDEX_DIR", "/data/processed/bank-index-v1"),
                         model_dir=os.getenv("E5_MODEL_DIR", "/data/models/e5-base"),
                         model_manifest=os.getenv("E5_MODEL_MANIFEST", "/srv/knowledge/embedding-model.json"),
                         allowed_bank_ids=None if os.getenv("BANK_ALLOWED_DOCUMENT_IDS", "*") == "*" else
                         frozenset(v.strip() for v in os.getenv("BANK_ALLOWED_DOCUMENT_IDS", "").split(",") if v.strip()))
            result.validate()
            return result
        except (ValueError, TypeError, KeyError, InvalidOperation):
            raise AnalysisError("invalid_llm_configuration") from None

    def validate(self):
        if self.budget_mode not in {"token_prices", "gateway_receipt"}:
            raise ValueError("invalid_budget_mode")
        if self.receipt_rub_per_unit is not None and (not self.receipt_rub_per_unit.is_finite() or not 0 < self.receipt_rub_per_unit <= 1_000_000):
            raise ValueError("invalid_receipt_unit")
        if not 0 < self.daily_limit_micros <= 500_000_000 or not 5 <= self.timeout_seconds <= 120:
            raise ValueError("invalid_limits")
        if not 3 <= self.max_calls <= 10 or not 1 <= self.max_tools <= 8:
            raise ValueError("invalid_agent_limits")
        for model in self.models.values():
            prices = (model.input_rub_per_million, model.output_rub_per_million)
            if (model.case_model not in CASE_MODELS or not model.id or len(model.id) > 160 or
                    any(p is not None and (not p.is_finite() or p < 0) for p in prices) or
                    (model.reasoning_enabled is not None and not isinstance(model.reasoning_enabled, bool)) or
                    model.response_mode not in {"text", "json_object", "json_schema"} or
                    not 4096 <= model.input_limit <= 262144 or not 256 <= model.output_limit <= 8192):
                raise ValueError("invalid_model_catalog")
        if self.base_url:
            parsed = urlsplit(self.base_url)
            if parsed.username or parsed.password or parsed.query or parsed.fragment or not parsed.hostname:
                raise ValueError("invalid_gateway_url")
            if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
                raise ValueError("gateway_requires_https")

    def require_ready(self):
        self.validate()
        if not self.base_url or not self.api_key or any(
                key not in self.models for key in (self.extraction_model, self.analysis_model)):
            raise AnalysisError("llm_not_configured")
        if self.budget_mode == "gateway_receipt":
            if self.receipt_rub_per_unit is None:
                raise AnalysisError("llm_not_configured")
        elif not self.prices_confirmed or any(
                self.models[key].input_rub_per_million is None or self.models[key].output_rub_per_million is None
                for key in (self.extraction_model, self.analysis_model)):
            raise AnalysisError("llm_not_configured")
