"""Chat-completions wire adapter with atomic, shared RUB accounting.

No HTTP retry: a timeout may have been billed. Unknown usage keeps its reserve.
The gateway protocol and actual model IDs must be confirmed by organizers.
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation

import httpx
from sqlalchemy import case, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .settings import AnalysisError, SchemaFailure, micros
from .storage import BudgetDay, CostReceipt, LLMCall


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def hashed(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def parse_json_object(content):
    # Some permitted models wrap an otherwise strict JSON response in one fence.
    # Accept the wrapper, never extract an object from arbitrary surrounding prose.
    value = content.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n(.*?)\n```", value, re.DOTALL)
    if fenced:
        value = fenced.group(1)

    def unique_object(pairs):
        result = {}
        for key, item in pairs:
            if key in result:
                raise ValueError("duplicate_json_key")
            result[key] = item
        return result

    def reject_constant(_):
        raise ValueError("non_json_number")

    try:
        result = json.loads(value, object_pairs_hook=unique_object, parse_constant=reject_constant)
    except (ValueError, TypeError, RecursionError):
        raise AnalysisError("llm_invalid_json") from None
    if not isinstance(result, dict):
        raise AnalysisError("llm_invalid_json")
    return result


def strict_schema(value):
    """Normalize optional/default fields to the strict structured-output subset."""
    if isinstance(value, list):
        return [strict_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {("anyOf" if key == "oneOf" else key): strict_schema(item)
              for key, item in value.items() if key not in {"default", "discriminator"}}
    if result.get("type") == "object" and "properties" in result:
        result["required"] = list(result["properties"])
        result["additionalProperties"] = False
    return result


def reserve(factory, settings, call_id, run_id, model, purpose, request_hash):
    day = datetime.now(timezone.utc).date().isoformat()
    receipt_mode = settings.budget_mode == "gateway_receipt"
    amount = None if receipt_mode else model.cost(model.input_limit, model.output_limit)
    with factory.begin() as db:
        insert = sqlite_insert if db.bind.dialect.name == "sqlite" else pg_insert
        db.execute(insert(BudgetDay).values(day=day, limit_micros=settings.daily_limit_micros,
                                           spent_micros=0, reserved_micros=0).on_conflict_do_nothing(index_elements=["day"]))
        # Always acquire the shared row's write lock before checking call identity
        # and the run cap. A conditional UPDATE that changes no row does not
        # serialize PostgreSQL workers with equal configured limits.
        db.execute(update(BudgetDay).where(BudgetDay.day == day).values(limit_micros=case(
            (BudgetDay.limit_micros > settings.daily_limit_micros, settings.daily_limit_micros),
            else_=BudgetDay.limit_micros)))
        previous = db.get(LLMCall, call_id)
        if previous:
            if previous.request_hash != request_hash:
                raise AnalysisError("resumed_call_context_changed")
            if previous.status == "completed" and previous.result is not None:
                return previous.result
            raise AnalysisError("previous_call_outcome_unknown")
        if db.scalar(select(func.count()).select_from(LLMCall).where(LLMCall.run_id == run_id)) >= settings.max_calls:
            raise AnalysisError("agent_call_limit")
        if receipt_mode:
            # Without tariffs, the entire remaining daily cap is the only safe upper bound.
            # Old unknown calls keep their reserves even if an operator explicitly
            # allocates more of the organizer's cap. Only the unreserved part is used.
            budget = db.scalar(select(BudgetDay).where(BudgetDay.day == day).with_for_update())
            amount = budget.limit_micros - budget.spent_micros - budget.reserved_micros
            if amount <= 0:
                raise AnalysisError("budget_exceeded")
            changed = db.execute(update(BudgetDay).where(
                BudgetDay.day == day, BudgetDay.reserved_micros == budget.reserved_micros,
                BudgetDay.spent_micros == budget.spent_micros, BudgetDay.limit_micros == budget.limit_micros,
            ).values(reserved_micros=BudgetDay.reserved_micros + amount))
            if changed.rowcount != 1:
                raise AnalysisError("budget_exceeded")
        else:
            changed = db.execute(update(BudgetDay).where(
                BudgetDay.day == day,
                BudgetDay.spent_micros + BudgetDay.reserved_micros + amount <= BudgetDay.limit_micros,
            ).values(reserved_micros=BudgetDay.reserved_micros + amount))
            if changed.rowcount != 1:
                raise AnalysisError("budget_exceeded")
        db.add(LLMCall(id=call_id, run_id=run_id, day=day, model=model.id, purpose=purpose,
                       request_hash=request_hash, status="reserved", reserved_micros=amount))
    return None


def settle(factory, call_id, model, result, usage, error=None, *, settings=None, receipt=None):
    with factory.begin() as db:
        # SQLite ignores FOR UPDATE. Claim by an atomic compare-and-swap so
        # simultaneous/repeated settlements cannot charge or release twice.
        # The temporary status is visible only inside this transaction; a crash
        # rolls back the claim together with every accounting change.
        changed = db.execute(update(LLMCall).where(LLMCall.id == call_id, LLMCall.status == "reserved")
                             .values(status="settling"))
        if changed.rowcount != 1:
            return
        call = db.get(LLMCall, call_id)
        incoming = usage.get("prompt_tokens") if isinstance(usage, dict) else None
        outgoing = usage.get("completion_tokens") if isinstance(usage, dict) else None
        valid = all(isinstance(v, int) and not isinstance(v, bool) and v >= 0 for v in (incoming, outgoing))
        receipt_mode = settings is not None and settings.budget_mode == "gateway_receipt"
        cost = None
        if receipt_mode:
            try:
                value = Decimal(receipt) if isinstance(receipt, str) and len(receipt) <= 80 else None
                if value is not None and value.is_finite() and 0 <= value <= 1_000_000:
                    cost = micros(value * settings.receipt_rub_per_unit)
                    db.add(CostReceipt(call_id=call_id, amount=str(value), rub_per_unit=str(settings.receipt_rub_per_unit),
                                       source="x-litellm-response-cost"))
            except (InvalidOperation, ValueError, OverflowError):
                pass
        elif valid:
            cost = model.cost(incoming, outgoing)
        if valid:
            call.input_tokens, call.output_tokens = incoming, outgoing
        if valid and cost is not None:
            call.cost_micros, call.input_tokens, call.output_tokens = cost, incoming, outgoing
            db.execute(update(BudgetDay).where(BudgetDay.day == call.day).values(
                reserved_micros=BudgetDay.reserved_micros - call.reserved_micros,
                spent_micros=BudgetDay.spent_micros + cost))
            if incoming > model.input_limit or outgoing > model.output_limit:
                error = "provider_exceeded_declared_token_limits"
            if cost > call.reserved_micros:
                error = "provider_exceeded_reserved_budget"
        call.result = result
        call.error = error or ("usage_unknown_reserve_retained" if not valid else
                               "cost_unknown_reserve_retained" if cost is None else None)
        call.status = "failed" if error else "completed" if valid and cost is not None else "usage_unknown" if not valid else "cost_unknown"


class Provider:
    def __init__(self, factory, settings, client=None):
        self.factory, self.settings = factory, settings
        self.client = client

    def generate(self, run_id, purpose, model_id, system, data, schema):
        self.settings.require_ready()
        if model_id not in self.settings.models:
            raise AnalysisError("model_not_allowed")
        model = self.settings.models[model_id]
        instruction = "\nReturn only a JSON object conforming to this schema:\n" + canonical(schema)
        if model.response_mode == "json_schema":
            instruction = "\nReturn only a JSON object conforming to the supplied response schema."
        messages = [{"role": "system", "content": system + instruction},
                    {"role": "user", "content": canonical({"untrusted_data": data})}]
        payload = {"model": model_id, "messages": messages, "max_tokens": model.output_limit, "stream": False}
        if model.response_mode == "json_schema":
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": schema.get("title", "Response"), "strict": True, "schema": strict_schema(schema)}}
        elif model.response_mode == "json_object":
            payload["response_format"] = {"type": "json_object"}
        # Include the native schema in the conservative UTF-8 input bound.
        if len(canonical(payload).encode()) + 256 > model.input_limit:
            raise AnalysisError("context_too_large")
        if model.reasoning_enabled is not None:
            payload["reasoning"] = {"enabled": model.reasoning_enabled}
        request_hash = hashed(payload)
        call_id = "llm_" + hashed([run_id, purpose])
        cached = reserve(self.factory, self.settings, call_id, run_id, model, purpose, request_hash)
        if cached is not None:
            return cached
        client = self.client or httpx.Client(timeout=httpx.Timeout(self.settings.timeout_seconds), follow_redirects=False)
        response_data = None
        receipt = None
        deadline = time.monotonic() + self.settings.timeout_seconds
        try:
            with client.stream("POST", self.settings.base_url + "/chat/completions", json=payload,
                               headers={"Authorization": "Bearer " + self.settings.api_key, "Accept": "application/json"}) as response:
                receipt = response.headers.get("x-litellm-response-cost")
                if time.monotonic() > deadline:
                    raise AnalysisError("llm_timeout_outcome_unknown")
                if response.status_code != 200:
                    raise AnalysisError("llm_http_" + str(response.status_code))
                body = bytearray()
                # Let HTTPX yield each available block. An 8 KiB chunker can
                # conceal a trickling response from the total-time check.
                for block in response.iter_bytes():
                    if time.monotonic() > deadline:
                        raise AnalysisError("llm_timeout_outcome_unknown")
                    body.extend(block)
                    if len(body) > 256_000:
                        raise AnalysisError("llm_response_too_large")
            response_data = json.loads(body)
            content = response_data["choices"][0]["message"]["content"]
            if not isinstance(content, str) or response_data["choices"][0].get("finish_reason") not in {None, "stop"}:
                raise AnalysisError("llm_incomplete_response")
            result = parse_json_object(content)
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
            settle(self.factory, call_id, model, result, usage, settings=self.settings, receipt=receipt)
            if not isinstance(usage, dict) or not all(isinstance(usage.get(k), int) and not isinstance(usage.get(k), bool)
                                                     and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                raise AnalysisError("llm_usage_missing")
            if usage["prompt_tokens"] > model.input_limit or usage["completion_tokens"] > model.output_limit:
                raise AnalysisError("provider_exceeded_declared_token_limits")
            with self.factory() as db:
                call = db.get(LLMCall, call_id)
                if call.status != "completed":
                    raise AnalysisError(call.error or "llm_cost_receipt_missing")
            return result
        finally:
            if self.client is None:
                client.close()
        settle(self.factory, call_id, model, None, response_data.get("usage") if isinstance(response_data, dict) else None,
               error, settings=self.settings, receipt=receipt)
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
        except ValueError as exc:
            issues = [".".join(map(str, e["loc"])) + ": " + e["msg"]
                      for e in exc.errors(include_input=False)][:12]
            raise SchemaFailure(issues) from None
