"""Read-only progress and an explicit, idempotent start/stop for autonomous collection."""
from datetime import datetime, timezone
import re
import time
from typing import Literal

from fastapi import Depends, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from app.core import Company
from app.news import Job, Event, new_id
from .storage import Insight
from .discovery import ACTIVE, MAX_SECONDS, DiscoveryRun, view_run
from .provider import hashed
from .service import company_profile
from .settings import AnalysisError


class DiscoveryRequest(BaseModel):
    model_config = ConfigDict(extra='forbid')
    target: int = Field(default=10, ge=1, le=10)
    min_grade: Literal[4, 5] = 4
    max_candidates: int = Field(default=40, ge=1, le=50)


def register_discovery_routes(app, get_db, current_session, settings_for):
    def own(run, demo):
        if run is None or run.company_id != demo.company_id or run.role != demo.role:
            raise HTTPException(404, 'Discovery run not found')
        return run

    @app.get('/api/v1/companies/{company_id}/discovery')
    def latest_discovery(company_id: str, demo=Depends(current_session), db=Depends(get_db)):
        if company_id != demo.company_id:
            raise HTTPException(403, 'Session does not grant access to this company')
        run = db.scalar(select(DiscoveryRun).where(DiscoveryRun.company_id == company_id,
            DiscoveryRun.role == demo.role).order_by(DiscoveryRun.created_at.desc(), DiscoveryRun.id.desc()).limit(1))
        return {'run': view_run(db, run) if run else None}

    @app.post('/api/v1/companies/{company_id}/discovery', status_code=202)
    def start_discovery(company_id: str, payload: DiscoveryRequest, idempotency_key: str | None = Header(None),
                        demo=Depends(current_session), db=Depends(get_db)):
        if company_id != demo.company_id:
            raise HTTPException(403, 'Session does not grant access to this company')
        if not idempotency_key or not re.fullmatch(r'[A-Za-z0-9._:-]{8,128}', idempotency_key):
            raise HTTPException(400, 'A valid Idempotency-Key is required')
        request_hash = hashed(payload.model_dump())
        previous = db.scalar(select(DiscoveryRun).where(DiscoveryRun.owner_session_hash == demo.token_hash,
                                                       DiscoveryRun.idempotency_key == idempotency_key))
        if previous:
            if previous.request_hash != request_hash:
                raise HTTPException(409, 'Idempotency-Key was already used with a different payload')
            return view_run(db, previous)
        active_key = company_id + ':' + demo.role
        active = db.scalar(select(DiscoveryRun).where(DiscoveryRun.active_key == active_key))
        if active:
            return view_run(db, active)
        settings = settings_for()
        try:
            settings.require_ready()
        except (AnalysisError, ValueError):
            raise HTTPException(503, 'llm_not_configured') from None
        company = db.get(Company, company_id)
        if company is None:
            raise HTTPException(404, 'Company not found')
        now = int(time.time())
        run = DiscoveryRun(id=new_id('search'), company_id=company_id, role=demo.role,
            owner_session_hash=demo.token_hash, idempotency_key=idempotency_key, request_hash=request_hash,
            active_key=active_key, status='collecting', stage='queued', target=payload.target,
            min_grade=payload.min_grade, max_candidates=payload.max_candidates,
            profile=company_profile(company, demo.role, db=db), items=[], collection=[], stats={}, feed_cursor=0,
            collection_lease_until=0, budget_day=datetime.now(timezone.utc).date().isoformat(),
            created_at=now, updated_at=now, deadline_at=now + MAX_SECONDS)
        try:
            with db.begin_nested():
                db.add(run)
                db.flush()
        except IntegrityError:
            active = db.scalar(select(DiscoveryRun).where(DiscoveryRun.active_key == active_key))
            if active:
                return view_run(db, active)
            raise HTTPException(409, 'Concurrent discovery request; read current progress') from None
        db.commit()
        return view_run(db, run)

    @app.get('/api/v1/discovery/{run_id}')
    def discovery_detail(run_id: str, demo=Depends(current_session), db=Depends(get_db)):
        return view_run(db, own(db.get(DiscoveryRun, run_id), demo))

    @app.post('/api/v1/discovery/{run_id}/stop')
    def stop_discovery(run_id: str, demo=Depends(current_session), db=Depends(get_db)):
        run = own(db.scalar(select(DiscoveryRun).where(DiscoveryRun.id == run_id).with_for_update()), demo)
        if run.status in ACTIVE:
            pending_ids = [j for item in run.items for j in (item.get('facts_job_id'), item.get('analysis_job_id')) if j]
            if pending_ids:
                db.execute(update(Job).where(Job.id.in_(pending_ids), Job.status == 'queued').values(
                    status='failed', stage='cancelled', error='discovery_cancelled', finished_at=int(time.time())))
            run.status, run.stage, run.stop_reason = 'stopping', 'stop_requested', 'user_cancelled'
            run.updated_at = int(time.time())
            db.commit()
        return view_run(db, run)

    @app.get('/api/v1/events/{event_id}/saved-insight')
    def saved_event_insight(event_id: str, demo=Depends(current_session), db=Depends(get_db)):
        event = db.get(Event, event_id)
        if event is None or event.company_id != demo.company_id:
            raise HTTPException(404, 'Event not found')
        insight = db.scalar(select(Insight).where(Insight.event_id == event_id,
            Insight.company_id == demo.company_id, Insight.role == demo.role)
            .order_by(Insight.created_at.desc(), Insight.id.desc()).limit(1))
        if insight is None:
            return {'insight': None}
        allowed = settings_for().allowed_bank_ids
        if allowed is not None and any(e.get('document_id') and e['document_id'] not in allowed
                                       for e in insight.evidence.values()):
            raise HTTPException(403, 'Analysis contains documents outside current access policy')
        from .service import public_insight_payload
        return {'insight': public_insight_payload(insight)}
