"""First backend increment: real company ETL + demo-scoped profile API.

No LLM or event-insight generation is claimed by these endpoints.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import time
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from app.core import Base, Company, DemoSession, feature_evidence, make_session_factory


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

    app = FastAPI(title="Alphich API", version="0.1.0", lifespan=lifespan)

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
                            "revenue_raw": c.revenue_raw, "revenue_unit": None,
                            "profile_version": c.profile_version} for c in records]}

    @app.get("/api/v1/companies/{company_id}/profile")
    def profile(company_id: str, demo: DemoSession = Depends(current_session), db: Session = Depends(get_db)):
        if company_id != demo.company_id:
            raise HTTPException(403, "Session does not grant access to this company")
        company = db.get(Company, company_id)
        if company is None:
            raise HTTPException(404, "Company profile not found")
        return {"id": company.id, "role": demo.role, "profile_version": company.profile_version,
                "industry": {"okved_code": company.okved_code, "okved_name": company.okved_name},
                "group_id": company.group_id, "region_registration": company.region,
                "city_registration": company.city, "revenue_raw": company.revenue_raw,
                "revenue_unit": None, "client_type": company.client_type,
                "products": company.products, "fts_flag": company.fts_flag,
                "fts_countries": company.countries, "features": feature_evidence(company),
                "quality_flags": company.quality_flags,
                "limitations": [
                    "This is an anonymized synthetic company profile.",
                    "Revenue currency and snapshot date have not been confirmed.",
                    "Registration region is not proof of geographic operations.",
                    "FTS countries do not establish trade direction or transaction currency.",
                    "Credit flag does not disclose debt amount, contract rate, or maturity.",
                ]}

    return app


app = create_app()
