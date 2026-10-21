"""Auditable ingestion of user-submitted and collected RSS publications.

This module deliberately DOES NOT extract news facts or call an LLM. Its job is
persisting input faithfully, grouping exact duplicates, and creating an event
that downstream extraction/analysis can enrich. Documents belong to their
recipient company; user-provided URLs are references, not downloads. The separate
RSS collector can fetch operator-configured sources and enqueue company copies.
"""
from __future__ import annotations

import hashlib
import ipaddress
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Literal
from urllib.parse import urlsplit, urlunsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import JSON, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from app.core import Base


class SourceDocument(Base):
    __tablename__ = "source_documents"
    __table_args__ = (
        UniqueConstraint("company_id", "content_hash", name="uq_source_company_hash"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    kind: Mapped[str] = mapped_column(String(40), nullable=False, default="user_submission")
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    source_url: Mapped[str | None] = mapped_column(String(2048))
    published_at: Mapped[str | None] = mapped_column(String(40))
    received_at: Mapped[str] = mapped_column(String(40), nullable=False)
    fetched_at: Mapped[str | None] = mapped_column(String(40))  # Null for pasted text; collector records actual fetch time.
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    extraction_status: Mapped[str] = mapped_column(String(48), nullable=False, default="raw_only")
    completeness: Mapped[str] = mapped_column(String(48), nullable=False, default="user_supplied_text")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class Event(Base):
    __tablename__ = "events"
    __table_args__ = (UniqueConstraint("source_id", name="uq_event_source"),)

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    source_id: Mapped[str] = mapped_column(String(40), nullable=False)
    title: Mapped[str] = mapped_column(String(300), nullable=False)
    published_at: Mapped[str | None] = mapped_column(String(40))
    event_at: Mapped[str | None] = mapped_column(String(40))  # Unknown until verified extraction.
    topics: Mapped[list[str]] = mapped_column(JSON, nullable=False, default=list)
    topic_method: Mapped[str] = mapped_column(String(40), nullable=False, default="not_classified")
    extraction_status: Mapped[str] = mapped_column(String(48), nullable=False, default="pending_extraction")
    completeness: Mapped[str] = mapped_column(String(48), nullable=False, default="user_supplied_text")
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)


class Job(Base):
    __tablename__ = "jobs"
    __table_args__ = (
        UniqueConstraint("owner_session_hash", "operation", "idempotency_key", name="uq_job_idempotency"),
        Index("ix_jobs_dispatch", "status", "available_at", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(40), primary_key=True)
    owner_session_hash: Mapped[str] = mapped_column(String(64), index=True)
    company_id: Mapped[str] = mapped_column(String(100), index=True)
    operation: Mapped[str] = mapped_column(String(40), nullable=False, default="publication_ingest")
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False)
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    source_id: Mapped[str] = mapped_column(String(40), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    stage: Mapped[str] = mapped_column(String(40), nullable=False, default="waiting")
    result_type: Mapped[str] = mapped_column(String(32), nullable=False, default="event")
    result_id: Mapped[str | None] = mapped_column(String(40))
    error: Mapped[str | None] = mapped_column(String(80))
    attempt_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[int] = mapped_column(Integer, nullable=False)
    available_at: Mapped[int] = mapped_column(Integer, nullable=False)
    lease_expires_at: Mapped[int | None] = mapped_column(Integer)
    heartbeat_at: Mapped[int | None] = mapped_column(Integer)
    finished_at: Mapped[int | None] = mapped_column(Integer)


class PublicationInput(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    title: str = Field(min_length=6, max_length=300)
    text: str = Field(min_length=30, max_length=100_000)
    source_url: str | None = Field(default=None, max_length=2048)
    published_at: datetime | None = None

    @field_validator("title", "text")
    @classmethod
    def meaningful_text(cls, v: str) -> str:
        # Avoid invisible/control characters while keeping ordinary newlines.
        v = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", v).strip()
        if len(v) < 6:
            raise ValueError("Text is empty after normalization")
        return v

    @field_validator("published_at")
    @classmethod
    def timezone_required(cls, v: datetime | None) -> datetime | None:
        if v is not None:
            if v.tzinfo is None or v.utcoffset() is None:
                raise ValueError("published_at must include an explicit timezone offset")
            return v.astimezone(timezone.utc)
        return v

    @field_validator("source_url")
    @classmethod
    def only_public_reference(cls, v: str | None) -> str | None:
        if not v:
            return None
        parts = urlsplit(v.strip())
        if parts.scheme not in {"https", "http"} or not parts.hostname or parts.username or parts.password:
            raise ValueError("source_url must be an http(s) URL without credentials")
        if parts.port is not None and parts.port not in (80, 443):
            raise ValueError("source_url has an unsupported port")
        host = parts.hostname.lower().rstrip(".")
        if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
            raise ValueError("source_url must not point to a local host")
        try:
            ip = ipaddress.ip_address(host.strip("[]"))
        except ValueError:
            pass  # A public-looking DNS name is only a reference here, not fetched.
        else:
            if not ip.is_global:
                raise ValueError("source_url must not point to a local or private IP")
        # No outbound request is made from the API or worker; URL is unverified.
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or "/", parts.query, ""))


class JobView(BaseModel):
    id: str
    status: Literal["queued", "running", "succeeded", "failed"]
    stage: str
    result_type: str
    result_id: str | None
    error: str | None


def view_job(job: Job) -> JobView:
    return JobView.model_validate({key: getattr(job, key) for key in JobView.model_fields})


def digest_input(payload: PublicationInput) -> str:
    # Exact normalized publication identity, including where/when it was sourced.
    # Different news agencies carrying the same text retain different provenance.
    canonical = {
        "title": " ".join(payload.title.split()),
        "text": " ".join(payload.text.split()),
        "source_url": payload.source_url,
        "published_at": payload.published_at.isoformat() if payload.published_at else None,
    }
    return hashlib.sha256(json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def digest_request(payload: PublicationInput) -> str:
    data = payload.model_dump(mode="json", exclude_none=False)
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return prefix + "_" + uuid.uuid4().hex


def serialize_event(event: Event, source: SourceDocument) -> dict:
    return {
        "id": event.id,
        "company_id": event.company_id,
        "version": event.version,
        "title": event.title,
        "published_at": event.published_at,
        "event_at": event.event_at,
        "topics": event.topics,
        "topic_method": event.topic_method,
        "facts": [],  # No LLM fact extraction is performed in this increment.
        "extraction_status": event.extraction_status,
        "completeness": event.completeness,
        "source_ids": [event.source_id],
        "sources": [{
            "id": source.id, "kind": source.kind, "title": source.title,
            "url": source.source_url, "url_verified": source.kind == "rss_article" and source.fetched_at is not None,
            "published_at": source.published_at, "received_at": source.received_at,
            "fetched_at": source.fetched_at,
            "content_hash": source.content_hash, "extraction_status": source.extraction_status,
            "completeness": source.completeness,
        }],
    }
