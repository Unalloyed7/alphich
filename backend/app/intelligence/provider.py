"""Chat-completions wire adapter with atomic, shared RUB accounting.

No HTTP retry: a timeout may have been billed. Unknown usage keeps its reserve.
The gateway protocol and actual model IDs must be confirmed by organizers.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import httpx
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .settings import AnalysisError
from .storage import BudgetDay, LLMCall


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hashed(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def reserve(factory, settings, call_id, run_id, model, purpose, request_hash):
    day = datetime.now(timezone.utc).date().isoformat()
    amount = model.cost(model.input_limit, model.output_limit)
    with factory.begin() as db:
        previous = db.get(LLMCall, call_id)
        if previous:
            if previous.request_hash != request_hash:
                raise AnalysisError("resumed_call_context_changed")
            if previous.status == "completed" and previous.result is not None:
                return previous.result
            raise AnalysisError("previous_call_outcome_unknown")
        insert = sqlite_insert if db.bind.dialect.name == "sqlite" else pg_insert
        db.execute(insert(BudgetDay).values(day=day, limit_micros=settings.daily_limit_micros,
                                           spent_micros=0, reserved_micros=0).on_conflict_do_nothing(index_elements=["day"]))
        if db.scalar(select(func.count()).select_from(LLMCall).where(LLMCall.run_id == run_id)) >= settings.max_calls:
            raise AnalysisError("agent_call_limit")
        # A second worker can lower the shared cap, but cannot raise it.
        db.execute(update(BudgetDay).where(BudgetDay.day == day, BudgetDay.limit_micros > settings.daily_limit_micros)
                   .values(limit_micros=settings.daily_limit_micros))
        changed = db.execute(update(BudgetDay).where(
            BudgetDay.day == day,
            BudgetDay.spent_micros + BudgetDay.reserved_micros + amount <= BudgetDay.limit_micros,
        ).values(reserved_micros=BudgetDay.reserved_micros + amount))
        if changed.rowcount != 1:
            raise AnalysisError("budget_exceeded")
        db.add(LLMCall(id=call_id, run_id=run_id, day=day, model=model.id, purpose=purpose,
                       request_hash=request_hash, status="reserved", reserved_micros=amount))
    return None


def settle(factory, call_id, model, result, usage, error=None):
    with factory.begin() as db:
        call = db.scalar(select(LLMCall).where(LLMCall.id == call_id).with_for_update())
        if call.status != "reserved":
            return
        incoming = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        outgoing = usage.get("completion_tokens") if isinstance(usage, dict) else None
        valid = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (incoming, outgoing))
        if valid:
            cost = model.cost(incoming, outgoing)
            call.cost_micros, call.input_tokens, call.output_tokens = cost, incoming, outgoing
            db.execute(update(BudgetDay).where(BudgetDay.day == call.day).values(
                reserved_micros=BudgetDay.reserved_micros - call.reserved_micros,
                spent_micros=BudgetDay.spent_micros + cost))
            if incoming > model.input_limit or outgoing > model.output_limit:
                error = "provider_exceeded_declared_token_limits"
        call.result = result
        call.error = error or (None if valid else "usage_unknown_reserve_retained")
        call.status = "failed" if error else "completed" if valid else "usage_unknown"


class Provider:
    def __init__(self, factory, settings, client=None):
        self.factory, self.settings = factory, settings
        self.client = client

    def generate(self, run_id, purpose, model_id, system, data, schema):
        self.settings.require_ready()
        if model_id not in self.settings.models:
            raise AnalysisError("model_not_allowed")
        model = self.settings.models[model_id]
        messages = [{"role": "system", "content": system + "\nReturn only a JSON object conforming to this schema:\n" + canonical(schema)},
                    {"role": "user", "content": canonical({"untrusted_data": data})}]
        # Conservative UTF-8 byte bound; no provider tokenizer or silent truncation.
        if len(canonical(messages).encode()) + 256 > model.input_limit:
            raise AnalysisError("context_too_large")
        payload = {"model": model_id, "messages": messages, "max_tokens": model.output_limit, "stream": False}
        request_hash = hashed(payload)
        call_id = "llm_" + hashed([run_id, purpose])
        cached = reserve(self.factory, self.settings, call_id, run_id, model, purpose, request_hash)
        if cached is not None:
            return cached
        client = self.client or httpx.Client(timeout=httpx.Timeout(self.settings.timeout_seconds), follow_redirects=False)
        response_data = None
        deadline = time.monotonic() + self.settings.timeout_seconds
        try:
            with client.stream("POST", self.settings.base_url + "/chat/completions", json=payload,
                               headers={"Authorization": "Bearer " + self.settings.api_key, "Accept": "application/json"}) as response:
                if time.monotonic() > deadline:
                    raise AnalysisError("llm_timeout_outcome_unknown")
                if response.status_code != 200:
                    raise AnalysisError("llm_http_" + str(response.status_code))
                body = bytearray()
                for block in response.iter_bytes(chunk_size=8192):
                    if time.monotonic() > deadline:
                        raise AnalysisError("llm_timeout_outcome_unknown")
                    body.extend(block)
                    if len(body) > 256_000:
                        raise AnalysisError("llm_response_too_large")
            response_data = json.loads(body)
            content = response_data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or response_data["choices"][0].get("finish_reason") not in {None, "stop"}:
                raise AnalysisError("llm_incomplete_response")
            result = json.loads(content)
            if not isinstance(result, dict):
                raise AnalysisError("llm_invalid_json")
        except httpx.TimeoutException:
            error = "llm_timeout_outcome_unknown"
        except httpx.HTTPError:
            error = "llm_network_outcome_unknown"
        except AnalysisError as exc:
            error = exc.code
        except (ValueError, KeyError, IndexError, TypeError, RecursionError):
            error = "llm_invalid_response"
        else:
            usage = response_data.get("usage")
            settle(self.factory, call_id, model, result, usage)
            if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool)
                                                     and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                raise AnalysisError("llm_usage_missing")
            if usage["prompt_tokens"] > model.input_limit or usage["completion_tokens"] > model.output_limit:
                raise AnalysisError("provider_exceeded_declared_token_limits")
            return result
        finally:
            if self.client is None:
                client.close()
        settle(self.factory, call_id, model, None, response_data.get("usage") if isinstance(response_data, dict) else None, error)
        raise AnalysisError(error)


class Calls:
    def __init__(self, provider, run_id, heartbeat=lambda: None):
        self.provider, self.run_id, self.heartbeat = provider, run_id, heartbeat
        self.count = 0

    def ask(self, purpose, schema, system, data, *, extraction=False):
        if self.count >= self.provider.settings.max_calls:
            raise AnalysisError("agent_call_limit")
        self.count += 1
        self.heartbeat()
        model = self.provider.settings.extraction_model if extraction else self.provider.settings.analysis_model
        result = self.provider.generate(self.run_id, purpose, model, system, data, schema.model_json_schema())
        try:
            return schema.model_validate(result)
        except ValueError:
            raise AnalysisError("llm_schema_invalid") from None
