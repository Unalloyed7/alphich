"""Opt-in labeling adapter using the gateway's verified daily budget ceiling.

An operator may transfer an old failed call's reservation to upstream budget
enforcement. Its cost and outcome remain unknown. No paid request is retried, and
ordinary receipt accounting still serializes and accounts for every new call.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

import httpx
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from .provider import Provider, hashed
from .settings import AnalysisError, micros
from .storage import BudgetDay, GatewayBudgetObservation, GatewayReserveRelease, LLMCall


def verified_budget(settings, client):
    if settings.budget_mode != "gateway_receipt" or settings.receipt_rub_per_unit is None:
        raise AnalysisError("gateway_guard_requires_receipt_currency")
    try:
        for attempt in range(3):
            try:
                response = client.get(settings.base_url.removesuffix("/v1") + "/key/info",
                                      headers={"Authorization": "Bearer " + settings.api_key})
            except httpx.HTTPError:
                if attempt == 2:
                    raise
                continue
            if response.status_code not in {408, 429, 500, 502, 503, 504} or attempt == 2:
                break
        # Only this read-only budget GET can retry. The paid POST below uses
        # the original Provider's single-dispatch, no-retry behavior.
        if response.status_code != 200:
            raise ValueError()
        info = response.json()["info"]
        if not isinstance(info, dict) or info.get("blocked") is True or info.get("status") not in (None, "active"):
            raise ValueError()
        values = [info.get(key) for key in ("max_budget", "spend")]
        if any(isinstance(v, bool) or not isinstance(v, (str, int, float)) for v in values):
            raise ValueError()
        ceiling, spend = map(lambda v: Decimal(str(v)), values)
        if not ceiling.is_finite() or not spend.is_finite() or not 0 <= spend <= ceiling or ceiling <= 0:
            raise ValueError()
        now = datetime.now(timezone.utc)
        end = datetime.fromisoformat(info["budget_reset_at"].replace("Z", "+00:00"))
        midnight = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        if info.get("budget_duration") != "1d" or end.tzinfo is None or end != midnight:
            raise ValueError()
        cap = micros(ceiling * settings.receipt_rub_per_unit)
        if not 0 < cap <= settings.daily_limit_micros <= 500_000_000:
            raise ValueError()
        return {"day": now.date().isoformat(), "observed_at": time.time_ns(),
                "gateway_fingerprint": hashlib.sha256(settings.base_url.encode()).hexdigest(),
                "key_fingerprint": hashlib.sha256(settings.api_key.encode()).hexdigest(),
                "native_spend": str(spend), "native_limit": str(ceiling),
                "rub_per_unit": str(settings.receipt_rub_per_unit), "reset_at": end.isoformat(),
                "spend_micros": micros(spend * settings.receipt_rub_per_unit), "limit_micros": cap}
    except (httpx.HTTPError, ValueError, KeyError, TypeError, AttributeError, InvalidOperation, OverflowError):
        raise AnalysisError("gateway_daily_budget_not_verified") from None


def observe_and_release(factory, snapshot, release_call_ids, *, failed_teacher_models=()):
    """Conservative aggregate spend floor; never assign aggregate cost to a call."""
    with factory.begin() as db:
        # Match Provider's budget-first locking order. Its full-cap reservation
        # still blocks concurrent workers while a new paid request is in flight.
        insert = sqlite_insert if db.bind.dialect.name == "sqlite" else pg_insert
        db.execute(insert(BudgetDay).values(day=snapshot["day"], limit_micros=snapshot["limit_micros"],
            spent_micros=0, reserved_micros=0).on_conflict_do_nothing(index_elements=["day"]))
        budget = db.scalar(select(BudgetDay).where(BudgetDay.day == snapshot["day"]).with_for_update())
        db.execute(update(BudgetDay).where(BudgetDay.day == budget.day)
                   .values(limit_micros=BudgetDay.limit_micros))
        db.refresh(budget)
        ceiling = min(budget.limit_micros, snapshot["limit_micros"])
        floor = max(budget.spent_micros, snapshot["spend_micros"])
        if floor >= ceiling:
            raise AnalysisError("budget_exceeded")
        observation_id = "gw_" + hashed(snapshot)
        observation = GatewayBudgetObservation(id=observation_id, day=budget.day,
            snapshot=snapshot, added_spend_micros=floor - budget.spent_micros)
        db.add(observation)
        released = 0
        ids = set(release_call_ids)
        if failed_teacher_models:
            ids.update(db.scalars(select(LLMCall.id).where(LLMCall.day == budget.day,
                LLMCall.model.in_(failed_teacher_models), LLMCall.purpose == "relevance_teacher_v2",
                LLMCall.status == "failed", LLMCall.cost_micros.is_(None), LLMCall.reserved_micros > 0,
                LLMCall.error.in_(("llm_timeout_outcome_unknown", "llm_network_outcome_unknown")))))
        for call_id in sorted(ids):
            if db.get(GatewayReserveRelease, call_id):
                continue
            call = db.get(LLMCall, call_id)
            if (not call or call.day != budget.day or call.status != "failed"
                    or call.error not in {"llm_timeout_outcome_unknown", "llm_network_outcome_unknown"}
                    or call.cost_micros is not None or call.reserved_micros <= 0):
                raise AnalysisError("gateway_guard_release_not_eligible")
            amount = call.reserved_micros
            claimed = db.execute(update(LLMCall).where(LLMCall.id == call_id,
                LLMCall.status == "failed", LLMCall.cost_micros.is_(None),
                LLMCall.reserved_micros == amount).values(reserved_micros=0))
            if claimed.rowcount != 1:
                raise AnalysisError("gateway_guard_release_changed")
            db.add(GatewayReserveRelease(call_id=call_id, observation_id=observation_id,
                released_micros=amount, reason="upstream_daily_cap_verified_cost_still_unknown"))
            released += amount
        if budget.reserved_micros < released:
            raise AnalysisError("gateway_guard_inconsistent_reserve")
        budget.reserved_micros -= released
        budget.spent_micros = floor
        budget.limit_micros = ceiling


class GuardedGatewayProvider(Provider):
    def __init__(self, factory, settings, client=None, *, release_call_ids=(), release_failed_teachers=False):
        super().__init__(factory, settings, client)
        self.release_call_ids = tuple(release_call_ids)
        self.failed_teacher_models = tuple(settings.models) if release_failed_teachers else ()

    def generate(self, *args, **kwargs):
        self.settings.require_ready()
        if self.client is not None:
            snapshot = verified_budget(self.settings, self.client)
        else:
            with httpx.Client(timeout=15, follow_redirects=False) as client:
                snapshot = verified_budget(self.settings, client)
        observe_and_release(self.factory, snapshot, self.release_call_ids,
                            failed_teacher_models=self.failed_teacher_models)
        return super().generate(*args, **kwargs)
