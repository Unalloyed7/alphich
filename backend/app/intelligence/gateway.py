"""Read-only organizer gateway diagnostics; never return the key-info payload."""
from __future__ import annotations

import os
import re
import shlex
from datetime import datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

from .settings import AnalysisError


def _budget_summary(info):
    """Copy only typed budget scalars, never arbitrary key-info objects/text."""
    if not isinstance(info, dict):
        raise ValueError("invalid_key_info")
    result = dict.fromkeys(("max_budget", "spend", "budget_duration", "budget_reset_at"))
    for key in ("max_budget", "spend"):
        value = info.get(key)
        if isinstance(value, (int, float, str)) and not isinstance(value, bool):
            try:
                number = Decimal(str(value))
                if number.is_finite() and number >= 0:
                    result[key] = value
            except (InvalidOperation, ValueError):
                pass
    duration = info.get("budget_duration")
    if isinstance(duration, str) and re.fullmatch(r"[0-9]{1,12}[a-z]{1,8}", duration):
        result["budget_duration"] = duration
    reset = info.get("budget_reset_at")
    if isinstance(reset, str) and len(reset) <= 80:
        try:
            datetime.fromisoformat(reset.replace("Z", "+00:00"))
            result["budget_reset_at"] = reset
        except ValueError:
            pass
    return result


def load_env_file(path):
    """Explicit CLI-only dotenv loading, without shell evaluation or interpolation."""
    try:
        values = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, value = line.removeprefix("export ").split("=", 1)
            key = key.strip()
            if not key.isidentifier():
                raise ValueError()
            parsed = shlex.split(value, comments=True)
            if len(parsed) > 1:
                raise ValueError()
            values[key] = parsed[0] if parsed else ""
        # An explicitly selected file is authoritative for this CLI invocation.
        os.environ.update(values)
    except (OSError, ValueError):
        raise AnalysisError("invalid_env_file") from None


def inspect_gateway(settings, client=None):
    if not settings.base_url or not settings.api_key:
        raise AnalysisError("llm_not_configured")
    own_client = client is None
    client = client or httpx.Client(timeout=15, follow_redirects=False)
    try:
        headers = {"Authorization": "Bearer " + settings.api_key}
        response = client.get(settings.base_url + "/models", headers=headers)
        if response.status_code != 200:
            raise AnalysisError("gateway_models_http_" + str(response.status_code))
        data = response.json()
        if not isinstance(data, dict) or not isinstance(data.get("data"), list):
            raise ValueError("invalid_model_list")
        available = {row["id"] for row in data["data"] if isinstance(row, dict) and isinstance(row.get("id"), str)}
        # Extra gateway models are deliberately excluded: case authorization is separate.
        models = sorted(available.intersection(settings.models))
        root = settings.base_url.removesuffix("/v1")
        response = client.get(root + "/key/info", headers=headers)
        budget = None
        if response.status_code == 200:
            data = response.json()
            if not isinstance(data, dict):
                raise ValueError("invalid_key_info")
            budget = _budget_summary(data.get("info", {}))
        return {"status": "reachable", "allowed_models_available": models,
                "selected_models_available": all(m in available for m in (settings.extraction_model, settings.analysis_model)),
                "upstream_budget": budget, "upstream_budget_unit": "gateway_native_not_implicitly_usd_or_rub",
                "budget_mode": settings.budget_mode}
    except httpx.HTTPError:
        raise AnalysisError("gateway_diagnostic_network_error") from None
    except (ValueError, KeyError, TypeError):
        raise AnalysisError("gateway_diagnostic_invalid_response") from None
    finally:
        if own_client:
            client.close()
