"""First backend increment: real company ETL + demo-scoped profile API.

No LLM or event-insight generation is claimed by these endpoints.
"""
from __future__ import annotations

import hashlib
import os
import re
import time
import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core import Base, Company, DemoSession, feature_evidence, make_session_factory
from alphich.organizer_semantics import revenue_metadata
from app.news import (Event, Job, JobView, PublicationInput, SourceDocument,
                      digest_input, digest_request, new_id, serialize_event, utc_iso, view_job)
from app.intelligence.routes import register_routes
from app.intelligence.service import event_payload
from app import rss as _rss  # Register additive collector tables in the shared metadata.
from app.news_collection import register_collection_routes
from app.rss_status import register_status_route


class DemoSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    company_id: str
    role: str


class DemoSessionResponse(BaseModel):
    token: str
    company_id: str
    role: str
    expires_at: int
    demo_only: bool = True


def create_app(session_factory=None, engine=None) -> FastAPI:
    if session_factory is None or engine is None:
        engine, session_factory = make_session_factory()

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        # P0 only. Replace create_all with migrations before expanding the schema.
        Base.metadata.create_all(engine)
        yield

    app = FastAPI(title="Alphich API", version="0.2.0", lifespan=lifespan)

    def get_db():
        with session_factory() as db:
            yield db

    def allowed_ids() -> set[str]:
        # Server-side allowlist, NOT user-supplied. Do not permit arbitrary company IDs.
        return {s.strip() for s in os.getenv("DEMO_COMPANY_IDS", "").split(",") if s.strip()}

    def current_session(x_demo_session: str | None = Header(default=None), db: Session = Depends(get_db)) -> DemoSession:
        if not x_demo_session:
            raise HTTPException(401, "Demo session required")
        token_hash = hashlib.sha256(x_demo_session.encode()).hexdigest()
        demo = db.get(DemoSession, token_hash)
        if demo is None or demo.expires_at <= int(time.time()):
            raise HTTPException(401, "Demo session expired or invalid")
        if demo.company_id not in allowed_ids():
            raise HTTPException(403, "Company is no longer available in demo")
        return demo

    @app.get("/api/v1/health")
    def health(db: Session = Depends(get_db)):
        try:
            db.execute(text("SELECT 1"))
        except Exception:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "Database unavailable")
        return {"status": "ok", "database": "ready"}

    @app.post("/api/v1/demo-session", response_model=DemoSessionResponse, status_code=201)
    def new_session(payload: DemoSessionRequest, db: Session = Depends(get_db)):
        if payload.role not in {"cfo", "ceo"}:
            raise HTTPException(422, "Supported roles: cfo, ceo")
        if payload.company_id not in allowed_ids():
            raise HTTPException(403, "Company is not on the server demo allowlist")
        if db.get(Company, payload.company_id) is None:
            raise HTTPException(404, "Company profile has not been imported")
        token = secrets.token_urlsafe(32)
        expires = int(time.time()) + 2 * 60 * 60
        db.add(DemoSession(token_hash=hashlib.sha256(token.encode()).hexdigest(),
                           company_id=payload.company_id, role=payload.role, expires_at=expires))
        db.commit()
        return DemoSessionResponse(token=token, company_id=payload.company_id,
                                   role=payload.role, expires_at=expires)

    @app.get("/api/v1/companies")
    def companies(_: DemoSession = Depends(current_session), db: Session = Depends(get_db)):
        ids = sorted(allowed_ids())
        if not ids:
            return {"items": []}
        records = db.scalars(select(Company).where(Company.id.in_(ids)).order_by(Company.id)).all()
        return {"items": [{"id": c.id, "okved_name": c.okved_name, "region": c.region,
                            "revenue_raw": revenue_metadata(c.revenue_raw)["amount"], "revenue_unit": "RUB",
                            "profile_version": c.profile_version} for c in records]}

    @app.get("/api/v1/companies/{company_id}/profile")
    def profile(company_id: str, demo: DemoSession = Depends(current_session), db: Session = Depends(get_db)):
        if company_id != demo.company_id:
            raise HTTPException(403, "Session does not grant access to this company")
        company = db.get(Company, company_id)
        if company is None:
            raise HTTPException(404, "Company profile not found")
        revenue = revenue_metadata(company.revenue_raw)
        return {"id": company.id, "role": demo.role, "profile_version": company.profile_version,
                "industry": {"okved_code": company.okved_code, "okved_name": company.okved_name},
                "group_id": company.group_id, "region_registration": company.region,
                "city_registration": company.city, "revenue_raw": revenue["amount"],
                "revenue_unit": revenue["unit"], "revenue_currency": revenue["currency"],
                "revenue_period": revenue["period"], "revenue_scope": revenue["scope"],
                "client_type": company.client_type,
                "products": company.products, "fts_flag": company.fts_flag,
                "fts_countries": company.countries, "features": feature_evidence(company),
                "quality_flags": company.quality_flags,
                "limitations": [
                    "This is an anonymized synthetic company profile.",
                    "Revenue is RUB for 2025; company/group scope and profile snapshot date are unknown.",
                    "Registration region is not proof of geographic operations.",
                    "FTS countries do not establish trade direction or transaction currency.",
                    "Credit flag does not disclose debt amount, contract rate, or maturity.",
                ]}

    @app.post("/api/v1/publications", status_code=202, response_model=JobView)
    def submit_publication(
        payload: PublicationInput,
        idempotency_key: str | None = Header(default=None),
        demo: DemoSession = Depends(current_session),
        db: Session = Depends(get_db),
    ):
        if not idempotency_key or not re.fullmatch(r"[A-Za-z0-9._:-]{8,128}", idempotency_key):
            raise HTTPException(400, "Idempotency-Key header must contain 8-128 URL-safe characters")
        if db.get(Company, demo.company_id) is None:
            raise HTTPException(403, "Selected company profile is no longer available")
        request_hash = digest_request(payload)
        old = db.scalar(select(Job).where(
            Job.owner_session_hash == demo.token_hash,
            Job.operation == "publication_ingest",
            Job.idempotency_key == idempotency_key,
        ))
        if old:
            if old.request_hash != request_hash:
                raise HTTPException(409, "Idempotency-Key was already used with a different payload")
            return view_job(old)

        # Atomic with job creation. Savepoints also allow recovery from the
        # unique-constraint race when several requests submit the same article.
        content_hash = digest_input(payload)
        source = db.scalar(select(SourceDocument).where(
            SourceDocument.company_id == demo.company_id,
            SourceDocument.content_hash == content_hash,
        ))
        if source is None:
            source = SourceDocument(
                id=new_id("src"), company_id=demo.company_id, title=payload.title,
                body=payload.text, source_url=payload.source_url,
                published_at=payload.published_at.isoformat() if payload.published_at else None,
                received_at=utc_iso(), fetched_at=None, content_hash=content_hash,
                extraction_status="raw_only", completeness="user_supplied_text", version=1,
            )
            try:
                with db.begin_nested():
                    db.add(source)
                    db.flush()
            except IntegrityError:
                source = db.scalar(select(SourceDocument).where(
                    SourceDocument.company_id == demo.company_id,
                    SourceDocument.content_hash == content_hash,
                ))
                if source is None:
                    raise

        now = int(time.time())
        job = Job(id=new_id("job"), owner_session_hash=demo.token_hash,
                  company_id=demo.company_id, operation="publication_ingest",
                  idempotency_key=idempotency_key, request_hash=request_hash,
                  source_id=source.id, status="queued", stage="waiting",
                  result_type="event", result_id=None, error=None, attempt_count=0,
                  created_at=now, available_at=now)
        try:
            with db.begin_nested():
                db.add(job)
                db.flush()
        except IntegrityError:
            job = db.scalar(select(Job).where(
                Job.owner_session_hash == demo.token_hash,
                Job.operation == "publication_ingest",
                Job.idempotency_key == idempotency_key,
            ))
            if job is None:
                raise
            if job.request_hash != request_hash:
                raise HTTPException(409, "Idempotency-Key was already used with a different payload")
        db.commit()
        return view_job(job)

    @app.get("/api/v1/jobs")
    def list_session_jobs(
        limit: int = Query(default=20, ge=1, le=50),
        operation: str = Query(default="publication_ingest", pattern="^(publication_ingest|insight_analysis|insight_question|news_fact_extraction|automatic_insight)$"),
        demo: DemoSession = Depends(current_session),
        db: Session = Depends(get_db),
    ):
        """Recent ingestion jobs belonging to this exact demo session.

        Deliberately does not broaden the existing per-session job ACL. The UI
        can recover history after reloading even if browser-side job IDs vanish.
        """
        rows = db.execute(
            select(Job, SourceDocument.title)
            .join(SourceDocument, Job.source_id == SourceDocument.id)
            .where(
                Job.owner_session_hash == demo.token_hash,
                Job.company_id == demo.company_id,
                SourceDocument.company_id == demo.company_id,
                Job.operation == operation,
            )
            .order_by(Job.created_at.desc(), Job.id.desc())
            .limit(limit)
        ).all()
        return {"items": [
            {**view_job(job).model_dump(), "created_at": job.created_at, "source_title": title}
            for job, title in rows
        ]}

    @app.get("/api/v1/jobs/{job_id}", response_model=JobView)
    def job_status(job_id: str, demo: DemoSession = Depends(current_session), db: Session = Depends(get_db)):
        job = db.get(Job, job_id)
        if job is None or job.owner_session_hash != demo.token_hash or job.company_id != demo.company_id:
            raise HTTPException(404, "Job not found")
        return view_job(job)

    @app.get("/api/v1/companies/{company_id}/events")
    def company_events(
        company_id: str,
        limit: int = Query(default=30, ge=1, le=50),
        offset: int = Query(default=0, ge=0, le=100_000),
        demo: DemoSession = Depends(current_session),
        db: Session = Depends(get_db),
    ):
        """List only completed raw events for the authorized demo company."""
        if company_id != demo.company_id:
            raise HTTPException(403, "Session does not grant access to this company")
        rows = db.execute(
            select(Event, SourceDocument)
            .join(SourceDocument, Event.source_id == SourceDocument.id)
            .where(Event.company_id == demo.company_id,
                   SourceDocument.company_id == demo.company_id)
            .order_by(SourceDocument.received_at.desc(), Event.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
        return {"items": [event_payload(db, event, source) for event, source in rows]}

    @app.get("/api/v1/events/{event_id}")
    def event_detail(event_id: str, demo: DemoSession = Depends(current_session), db: Session = Depends(get_db)):
        event = db.get(Event, event_id)
        if event is None or event.company_id != demo.company_id:
            raise HTTPException(404, "Event not found")
        source = db.get(SourceDocument, event.source_id)
        if source is None or source.company_id != demo.company_id:
            raise HTTPException(404, "Source not found")
        return event_payload(db, event, source)

    @app.get("/api/v1/sources/{source_id}")
    def source_detail(source_id: str, demo: DemoSession = Depends(current_session), db: Session = Depends(get_db)):
        source = db.get(SourceDocument, source_id)
        if source is None or source.company_id != demo.company_id:
            raise HTTPException(404, "Source not found")
        collected = source.kind == "rss_article" and source.fetched_at is not None
        return {"id": source.id, "company_id": source.company_id, "title": source.title, "kind": source.kind,
                "text": source.body, "url": source.source_url, "url_verified": collected,
                "published_at": source.published_at, "received_at": source.received_at,
                "fetched_at": source.fetched_at,
                "content_hash": source.content_hash, "version": source.version,
                "extraction_status": source.extraction_status, "completeness": source.completeness,
                "note": ("Article text extracted from a fetched RSS link. Fetching does not verify factual accuracy."
                         if collected else "User-submitted text. URL is not fetched or independently verified.")}

    register_collection_routes(app, get_db, current_session)
    register_status_route(app, get_db, current_session)
    register_routes(app, get_db, current_session, session_factory)
    return app


app = create_app()
