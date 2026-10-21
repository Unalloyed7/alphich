"""Session-scoped analysis jobs, feed, evidence, conversations and public audit."""
from __future__ import annotations

import re
import time
from datetime import datetime, timezone

from fastapi import Depends, Header, HTTPException, Query
from sqlalchemy import or_, select
from sqlalchemy.exc import IntegrityError

from app.core import Company
from app.news import Event, Job, JobView, SourceDocument, new_id, view_job
from . import AGENT_ROLES
from .provider import call_role, hashed
from .schemas import InsightRequest, QuestionRequest
from .service import company_profile, normalized_event
from .settings import AnalysisError, Settings
from .gateway import inspect_gateway
from .storage import AnalysisRun, AnalysisTask, Answer, BudgetDay, CostReceipt, Insight, LLMCall
from . import news_index as _news_index  # Register additive persistent index tables at startup.
from .automation import AutomationRecord, AutomationConfig


def register_routes(app, get_db, current_session, factory):
    def config():
        try:
            return Settings.from_env()
        except AnalysisError as exc:
            raise HTTPException(503, exc.code) from None

    from .discovery_routes import register_discovery_routes
    register_discovery_routes(app, get_db, current_session, config)

    def authorized(record, demo):
        if record is None or record.company_id != demo.company_id or record.role != demo.role:
            raise HTTPException(404, "Analysis result not found")
        return record

    def allowed_evidence(evidence, settings):
        if settings.allowed_bank_ids is not None and any(
                e.get("document_id") and e["document_id"] not in settings.allowed_bank_ids for e in evidence.values()):
            raise HTTPException(403, "Analysis contains documents outside current access policy")

    def enqueue(db, demo, event, source, payload, key, profile, parent=None, question=None, *, facts_only=False):
        settings = config()
        try:
            settings.require_ready()
        except (AnalysisError, ValueError):
            raise HTTPException(503, "llm_not_configured") from None
        if not key or not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", key):
            raise HTTPException(400, "A valid Idempotency-Key is required")
        operation = "news_fact_extraction" if facts_only else "insight_question" if parent else "insight_analysis"
        request_hash = hashed(payload)
        old = db.scalar(select(Job).where(Job.owner_session_hash == demo.token_hash, Job.operation == operation,
                                         Job.idempotency_key == key))
        if old:
            if old.request_hash != request_hash:
                raise HTTPException(409, "Idempotency-Key was already used with a different payload")
            return view_job(old)
        now = int(time.time())
        job = Job(id=new_id("job"), owner_session_hash=demo.token_hash, company_id=demo.company_id,
                  operation=operation, idempotency_key=key, request_hash=request_hash, source_id=source.id,
                  status="queued", stage="waiting_facts" if facts_only else "waiting_analysis",
                  result_type="event" if facts_only else "answer" if parent else "insight",
                  result_id=None, attempt_count=0, created_at=now, available_at=now)
        as_of = parent.payload["as_of"] if parent else payload.get("as_of") or datetime.now(timezone.utc).isoformat()
        task = AnalysisTask(job_id=job.id, event_id=event.id, role=demo.role, as_of=as_of, profile=profile,
                            parent_insight_id=parent.id if parent else None, question=question)
        try:
            with db.begin_nested():
                db.add_all([job, task])
                db.flush()
        except IntegrityError:
            old = db.scalar(select(Job).where(Job.owner_session_hash == demo.token_hash, Job.operation == operation,
                                             Job.idempotency_key == key))
            if old is None:
                raise
            if old.request_hash != request_hash:
                raise HTTPException(409, "Idempotency-Key was already used with a different payload")
            return view_job(old)
        db.commit()
        return view_job(job)

    @app.get("/api/v1/llm/status")
    def llm_status(demo=Depends(current_session), db=Depends(get_db)):
        settings = config()
        ready = True
        try:
            settings.require_ready()
        except (AnalysisError, ValueError):
            ready = False
        day = datetime.now(timezone.utc).date().isoformat()
        budget = db.get(BudgetDay, day)
        return {"configured": ready, "protocol": "chat_completions_http", "prices_confirmed": settings.prices_confirmed,
                "agent_network": {"roles": list(AGENT_ROLES), "max_calls": settings.max_calls, "max_tools": settings.max_tools},
                "budget_mode": settings.budget_mode,
                "receipt_rub_per_unit": str(settings.receipt_rub_per_unit) if settings.receipt_rub_per_unit is not None else None,
                "daily_limit_rub": str(min(settings.daily_limit_micros, budget.limit_micros if budget else settings.daily_limit_micros) / 1_000_000), "budget_day_utc": day,
                "spent_rub": str((budget.spent_micros if budget else 0) / 1_000_000),
                "reserved_rub": str((budget.reserved_micros if budget else 0) / 1_000_000),
                "extraction_model": settings.extraction_model or None, "analysis_model": settings.analysis_model or None,
                "bank_index_present": __import__("pathlib").Path(settings.bank_index).is_dir()}

    @app.get("/api/v1/llm/gateway-status")
    def gateway_status(demo=Depends(current_session)):
        # GET /models and /key/info only; never a completion or ledger mutation.
        try:
            return inspect_gateway(config())
        except AnalysisError as exc:
            raise HTTPException(503, exc.code) from None

    @app.get("/api/v1/companies/{company_id}/automation")
    def automation_status(company_id: str, demo=Depends(current_session), db=Depends(get_db)):
        if company_id != demo.company_id:
            raise HTTPException(403, "Session does not grant access to this company")
        rows = db.scalars(select(AutomationRecord).where(AutomationRecord.company_id == company_id,
            AutomationRecord.role == demo.role).order_by(AutomationRecord.updated_at.desc(), AutomationRecord.id).limit(50)).all()
        return {"enabled": AutomationConfig.from_env().enabled, "company_id": company_id, "role": demo.role,
            "items": [{"event_id": row.event_id, "status": row.status, "grade": row.grade,
                "as_of": row.as_of, "error": row.error, "assessment": row.assessment,
                "extraction_job_id": row.extraction_job_id, "analysis_job_id": row.analysis_job_id} for row in rows]}

    @app.post("/api/v1/companies/{company_id}/insights", status_code=202, response_model=JobView)
    def create_insight(company_id: str, payload: InsightRequest, idempotency_key: str | None = Header(None),
                       demo=Depends(current_session), db=Depends(get_db)):
        if company_id != demo.company_id:
            raise HTTPException(403, "Session does not grant access to this company")
        event = db.get(Event, payload.event_id)
        source = db.get(SourceDocument, event.source_id) if event else None
        company = db.get(Company, company_id)
        if not event or not source or not company or event.company_id != company_id or source.company_id != company_id:
            raise HTTPException(404, "Event not found")
        return enqueue(db, demo, event, source, payload.model_dump(mode="json"), idempotency_key, company_profile(company, demo.role))

    @app.post("/api/v1/events/{event_id}/extract-facts", status_code=202, response_model=JobView)
    def extract_event_facts(event_id: str, idempotency_key: str | None = Header(None),
                            demo=Depends(current_session), db=Depends(get_db)):
        """Billable extraction only; reuse the worker/cache/budget, never create a personal card."""
        event = db.get(Event, event_id)
        source = db.get(SourceDocument, event.source_id) if event else None
        company = db.get(Company, demo.company_id)
        if not event or not source or not company or event.company_id != demo.company_id or source.company_id != demo.company_id:
            raise HTTPException(404, "Event not found")
        return enqueue(db, demo, event, source, {"event_id": event_id}, idempotency_key,
                       company_profile(company, demo.role), facts_only=True)

    @app.get("/api/v1/insights/{insight_id}")
    def insight_detail(insight_id: str, demo=Depends(current_session), db=Depends(get_db)):
        insight = authorized(db.get(Insight, insight_id), demo)
        allowed_evidence(insight.evidence, config())
        return insight.payload

    @app.post("/api/v1/insights/{insight_id}/questions", status_code=202, response_model=JobView)
    def question(insight_id: str, payload: QuestionRequest, idempotency_key: str | None = Header(None),
                 demo=Depends(current_session), db=Depends(get_db)):
        parent = authorized(db.get(Insight, insight_id), demo)
        allowed_evidence(parent.evidence, config())
        event = db.get(Event, parent.event_id)
        source = db.get(SourceDocument, event.source_id) if event else None
        if not event or not source or event.company_id != demo.company_id or source.company_id != demo.company_id:
            raise HTTPException(404, "Event not found")
        return enqueue(db, demo, event, source, {"insight_id": insight_id, **payload.model_dump()},
                       idempotency_key, parent.snapshot["profile"], parent, payload.question)

    @app.get("/api/v1/answers/{answer_id}")
    def answer_detail(answer_id: str, demo=Depends(current_session), db=Depends(get_db)):
        answer = authorized(db.get(Answer, answer_id), demo)
        allowed_evidence(answer.evidence, config())
        return answer.payload

    @app.get("/api/v1/runs/{run_id}")
    def run_detail(run_id: str, demo=Depends(current_session), db=Depends(get_db)):
        run = authorized(db.get(AnalysisRun, run_id), demo)
        settings = config()
        if settings.allowed_bank_ids is not None and not set(run.versions.get("allowed_bank_ids", [])) <= settings.allowed_bank_ids:
            raise HTTPException(403, "Run contains documents outside current access policy")
        calls = db.scalars(select(LLMCall).where(LLMCall.run_id == run_id)).all()
        receipts = {r.call_id: r for r in db.scalars(select(CostReceipt).where(CostReceipt.call_id.in_([c.id for c in calls])))}
        return {"id": run.id, "company_id": run.company_id, "role": run.role, "as_of": run.as_of, "status": run.status,
                "steps": run.steps, "versions": run.versions, "verification": run.verification,
                "llm_calls": [{"id": c.id, "model": c.model, "purpose": c.purpose, "agent_role": call_role(c.purpose), "status": c.status,
                               "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                               "cost_rub": str(c.cost_micros / 1_000_000) if c.cost_micros is not None else None,
                               "cost_source": receipts[c.id].source if c.id in receipts else "configured_token_prices",
                               "receipt": {"amount": receipts[c.id].amount, "rub_per_unit": receipts[c.id].rub_per_unit,
                                           "source": receipts[c.id].source} if c.id in receipts else None,
                               "reserved_rub": str(c.reserved_micros / 1_000_000), "error": c.error} for c in calls]}

    @app.get("/api/v1/evidence/{evidence_id}")
    def evidence_detail(evidence_id: str, insight_id: str = Query(...), answer_id: str | None = None,
                        demo=Depends(current_session), db=Depends(get_db)):
        parent = authorized(db.get(Insight, insight_id), demo)
        result = authorized(db.get(Answer, answer_id), demo) if answer_id else parent
        if answer_id and result.insight_id != parent.id:
            raise HTTPException(404, "Evidence not found")
        evidence = result.evidence.get(evidence_id)
        if evidence is None:
            raise HTTPException(404, "Evidence not found")
        allowed_evidence({evidence_id: evidence}, config())
        return evidence

    @app.get("/api/v1/companies/{company_id}/feed")
    def feed(company_id: str, as_of: datetime | None = None, demo=Depends(current_session), db=Depends(get_db)):
        from alphich_kb.personalization.inputs import SelectionScope
        from alphich_kb.personalization.selector import NewsSelector
        from alphich_kb.personalization.grading import MIN_SELECTED_GRADE
        if company_id != demo.company_id:
            raise HTTPException(403, "Session does not grant access to this company")
        try:
            when = InsightRequest(event_id="scope", as_of=as_of).as_of or datetime.now(timezone.utc)
        except ValueError:
            raise HTTPException(422, "A timezone-aware as_of at or before now is required") from None
        company = db.get(Company, company_id)
        if company is None:
            raise HTTPException(404, "Company not found")
        settings = config()
        rows = db.execute(select(Event, SourceDocument).join(SourceDocument, Event.source_id == SourceDocument.id).where(
            Event.company_id == company_id, SourceDocument.company_id == company_id,
            SourceDocument.received_at <= when.isoformat(),
            or_(SourceDocument.published_at.is_(None), SourceDocument.published_at <= when.isoformat()))
            .order_by(SourceDocument.received_at.desc(), Event.id.desc()).limit(settings.news_feed_limit)).all()
        events = [normalized_event(db, e, s) for e, s in rows]
        from .news_index import CachedNewsEncoder, encoder_for
        encoder = encoder_for(settings)
        cached = CachedNewsEncoder(factory, encoder, company_id) if encoder is not None else None
        result = NewsSelector(events, cached).select(company_profile(company, demo.role),
            SelectionScope(company_id, when.isoformat(), frozenset(e["id"] for e in events), demo.role),
            mode="hybrid" if cached else "lexical", limit=10, min_grade=MIN_SELECTED_GRADE)
        recent = db.scalars(select(Insight).where(Insight.company_id == company_id, Insight.role == demo.role)
                            .order_by(Insight.created_at.desc(), Insight.id.desc()).limit(100)).all()
        for item in result["items"]:
            match = next((r for r in recent if r.event_id == item["event_id"] and
                          r.payload["event_version"] == item["event_version"] and
                          r.payload["profile_version"] == company.profile_version and
                          datetime.fromisoformat(r.payload["as_of"]) <= when), None)
            if match:
                try:
                    allowed_evidence(match.evidence, config())
                except HTTPException:
                    continue
                item["insight"] = {k: match.payload[k] for k in ("id", "status", "as_of", "headline", "bank_view")}
        result["limitations"] = ["bounded_candidate_pool", "machine_relevance_not_human_review"]
        if not cached or result["mode"] == "lexical_fallback":
            result["limitations"].append("news_dense_index_unavailable_or_pending")
        result["candidate_pool_limit"] = settings.news_feed_limit
        result["news_index"] = {"available": bool(cached and result["mode"] == "hybrid"),
            "space_id": encoder.space_id if encoder else None, "passage_cache_hits": cached.hits if cached else 0,
            "passage_cache_misses": cached.misses if cached else 0, "builds_on_read": False}
        return result
