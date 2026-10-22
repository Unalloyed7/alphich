"""Additive tables: immutable outputs, task inputs, global budget and call audit."""
from sqlalchemy import JSON, BigInteger, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core import Base


class BudgetDay(Base):
    __tablename__ = "llm_budget_days"
    day: Mapped[str] = mapped_column(String(10), primary_key=True)
    limit_micros: Mapped[int] = mapped_column(BigInteger)
    spent_micros: Mapped[int] = mapped_column(BigInteger, default=0)
    reserved_micros: Mapped[int] = mapped_column(BigInteger, default=0)


class LLMCall(Base):
    __tablename__ = "llm_calls"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40), index=True)
    day: Mapped[str] = mapped_column(String(10), index=True)
    model: Mapped[str] = mapped_column(String(160))
    purpose: Mapped[str] = mapped_column(String(48))
    request_hash: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(40))
    reserved_micros: Mapped[int] = mapped_column(BigInteger)
    cost_micros: Mapped[int | None] = mapped_column(BigInteger)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    result: Mapped[dict | None] = mapped_column(JSON)
    error: Mapped[str | None] = mapped_column(String(80))


class CostReceipt(Base):
    __tablename__ = "llm_cost_receipts"
    call_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    amount: Mapped[str] = mapped_column(String(100))
    rub_per_unit: Mapped[str] = mapped_column(String(100))
    source: Mapped[str] = mapped_column(String(80))


class CostReconciliation(Base):
    __tablename__ = "llm_cost_reconciliations"
    call_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    reason: Mapped[str] = mapped_column(String(80))
    released_micros: Mapped[int] = mapped_column(BigInteger)
    created_at: Mapped[int] = mapped_column(BigInteger)


class AnalysisRun(Base):
    __tablename__ = "analysis_runs"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    role: Mapped[str] = mapped_column(String(8))
    as_of: Mapped[str] = mapped_column(String(40))
    status: Mapped[str] = mapped_column(String(40))
    steps: Mapped[list] = mapped_column(JSON, default=list)
    versions: Mapped[dict] = mapped_column(JSON, default=dict)
    verification: Mapped[dict] = mapped_column(JSON, default=dict)
    created_at: Mapped[int] = mapped_column(BigInteger)


class Extraction(Base):
    __tablename__ = "event_extractions"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    source_id: Mapped[str] = mapped_column(String(40), index=True)
    source_version: Mapped[int] = mapped_column(Integer)
    event_id: Mapped[str] = mapped_column(String(40), index=True)
    event_version: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(40))
    facts: Mapped[list] = mapped_column(JSON, default=list)
    limitations: Mapped[list] = mapped_column(JSON, default=list)
    coverage: Mapped[dict] = mapped_column(JSON, default=dict)
    run_id: Mapped[str] = mapped_column(String(40))
    lease_expires_at: Mapped[int] = mapped_column(BigInteger, default=0)


class PublicArticleExtraction(Base):
    """Only collector-proven public article facts; never company prompts/results."""
    __tablename__ = "public_article_extractions"
    id: Mapped[str] = mapped_column(String(80), primary_key=True)
    article_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(40))
    facts: Mapped[list] = mapped_column(JSON, default=list)
    coverage: Mapped[dict] = mapped_column(JSON, default=dict)
    limitations: Mapped[list] = mapped_column(JSON, default=list)
    run_id: Mapped[str] = mapped_column(String(40))
    lease_expires_at: Mapped[int] = mapped_column(BigInteger)


class AnalysisTask(Base):
    __tablename__ = "analysis_tasks"
    job_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(40))
    role: Mapped[str] = mapped_column(String(8))
    as_of: Mapped[str] = mapped_column(String(40))
    profile: Mapped[dict] = mapped_column(JSON)
    parent_insight_id: Mapped[str | None] = mapped_column(String(40))
    question: Mapped[str | None] = mapped_column(Text)


class AnalysisLease(Base):
    __tablename__ = "analysis_cache_leases"
    cache_key: Mapped[str] = mapped_column(String(64), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(40))
    expires_at: Mapped[int] = mapped_column(BigInteger)


class Insight(Base):
    __tablename__ = "insights"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    role: Mapped[str] = mapped_column(String(8))
    event_id: Mapped[str] = mapped_column(String(40), index=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True)
    payload: Mapped[dict] = mapped_column(JSON)
    evidence: Mapped[dict] = mapped_column(JSON)
    snapshot: Mapped[dict] = mapped_column(JSON)
    run_id: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[int] = mapped_column(BigInteger)


class Answer(Base):
    __tablename__ = "insight_answers"
    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    insight_id: Mapped[str] = mapped_column(String(40), index=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    role: Mapped[str] = mapped_column(String(8))
    payload: Mapped[dict] = mapped_column(JSON)
    evidence: Mapped[dict] = mapped_column(JSON)
    run_id: Mapped[str] = mapped_column(String(40))
    created_at: Mapped[int] = mapped_column(BigInteger)


class InsightFeedback(Base):
    """Latest explicit review per demo session/card, with server-owned provenance."""
    __tablename__ = "insight_feedback"
    session_hash: Mapped[str] = mapped_column(String(64), primary_key=True)
    insight_id: Mapped[str] = mapped_column(String(40), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    role: Mapped[str] = mapped_column(String(8))
    value: Mapped[str] = mapped_column(String(24))
    comment: Mapped[str] = mapped_column(String(500))
    context: Mapped[dict] = mapped_column(JSON)
    created_at: Mapped[int] = mapped_column(BigInteger)
    updated_at: Mapped[int] = mapped_column(BigInteger)
