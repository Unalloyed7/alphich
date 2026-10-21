"""Public RSS fact reuse; binding to tenant sources happens after the shared stage."""
from copy import deepcopy
import time

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.news import Event, SourceDocument
from .extraction import EXTRACTION_VERSION, extract
from .provider import hashed
from .settings import AnalysisError
from .storage import Extraction, LLMCall, PublicArticleExtraction


def _configuration(settings):
    return [EXTRACTION_VERSION, *[[m.id, m.input_limit, m.output_limit, m.reasoning_enabled, m.response_mode]
        for m in (settings.models[settings.extraction_model], settings.models[settings.analysis_model])]]


def _public_snapshot(db, source):
    from app.rss import RssArticle, RssDelivery
    if source.kind != "rss_article":
        return None
    article = db.scalar(select(RssArticle).join(RssDelivery, RssDelivery.article_id == RssArticle.id).where(
        RssDelivery.source_id == source.id, RssDelivery.company_id == source.company_id))
    if article is None or any((source.body != article.body, source.title != article.title,
                              source.content_hash != article.content_hash, source.source_url != article.final_url,
                              source.published_at != article.published_at, source.fetched_at != article.fetched_at)):
        return None
    return {"id": "public_article_" + article.id, "article_id": article.id, "version": 1,
            "title": article.title, "text": article.body, "received_at": article.fetched_at,
            "published_at": article.published_at}


def _bind(facts, snapshot):
    result = []
    for raw in facts:
        fact = deepcopy(raw)
        start, end = fact["source_span"]["start"], fact["source_span"]["end"]
        if not 0 <= start < end <= len(snapshot["text"]) or snapshot["text"][start:end] != fact["text"]:
            raise AnalysisError("shared_extraction_quote_mismatch")
        fact.update(source_id=snapshot["id"], source_version=str(snapshot["version"]),
                    source_available_at=snapshot["received_at"])
        fact.pop("id", None)
        fact["id"] = "event_fact_" + hashed(fact)
        result.append(fact)
    return result


def extract_for_event(runtime, event_id, calls):
    factory = runtime.factory
    config = _configuration(runtime.settings)
    with factory() as db:
        event = db.get(Event, event_id)
        source = db.get(SourceDocument, event.source_id) if event else None
        if not source or source.company_id != event.company_id:
            raise AnalysisError("event_outside_scope")
        key = "extraction_" + hashed([source.id, source.version, source.content_hash, config])
        existing = db.get(Extraction, key)
        if existing and existing.status == "completed":
            return key
        if existing:
            raise AnalysisError("extraction_in_progress" if existing.status == "running" and
                existing.lease_expires_at > int(time.time()) else "extraction_outcome_unknown")
        unresolved = db.scalar(select(LLMCall.id).join(Extraction, Extraction.run_id == LLMCall.run_id).where(
            Extraction.source_id == source.id, Extraction.status == "failed", LLMCall.cost_micros.is_(None)).limit(1))
        if unresolved:
            raise AnalysisError("extraction_outcome_unknown")
        snapshot = {"id": source.id, "version": source.version, "title": source.title,
                    "text": source.body, "received_at": source.received_at}
        public = _public_snapshot(db, source)
    shared_key = "public_" + hashed([public, config]) if public else None
    shared_owned = False
    if shared_key:
        try:
            with factory.begin() as db:
                unknown = db.scalar(select(LLMCall.id).join(PublicArticleExtraction,
                    PublicArticleExtraction.run_id == LLMCall.run_id).where(
                    PublicArticleExtraction.article_id == public["article_id"],
                    PublicArticleExtraction.status == "failed", LLMCall.cost_micros.is_(None)).limit(1))
                if unknown:
                    raise AnalysisError("extraction_outcome_unknown")
                shared = db.get(PublicArticleExtraction, shared_key)
                if shared is None:
                    db.add(PublicArticleExtraction(id=shared_key, article_id=public["article_id"],
                        status="running", run_id=calls.run_id, lease_expires_at=int(time.time())+600))
                    shared_owned = True
                elif shared.status != "completed":
                    raise AnalysisError("extraction_in_progress" if shared.status == "running" and
                        shared.lease_expires_at > int(time.time()) else "extraction_outcome_unknown")
                else:
                    facts, coverage, limits = deepcopy(shared.facts), deepcopy(shared.coverage), list(shared.limitations)
        except IntegrityError:
            raise AnalysisError("extraction_in_progress") from None
    try:
        with factory.begin() as db:
            db.add(Extraction(id=key, source_id=snapshot["id"], source_version=snapshot["version"],
                event_id=event_id, event_version=0, status="running", run_id=calls.run_id,
                lease_expires_at=int(time.time())+600))
    except IntegrityError:
        if shared_owned:
            with factory.begin() as db:
                db.delete(db.get(PublicArticleExtraction, shared_key))  # No request was sent.
        raise AnalysisError("extraction_in_progress") from None
    try:
        if not shared_key or shared_owned:
            facts, coverage, limits = extract(public or snapshot, calls)
        calls.heartbeat()
        bound = _bind(facts, snapshot) if public else facts
        with factory.begin() as db:
            event = db.scalar(select(Event).where(Event.id == event_id).with_for_update())
            source = db.scalar(select(SourceDocument).where(SourceDocument.id == snapshot["id"]).with_for_update())
            record = db.get(Extraction, key)
            if record.run_id != calls.run_id or record.status != "running":
                raise AnalysisError("extraction_lease_lost")
            if source.version != snapshot["version"] or source.body != snapshot["text"]:
                raise AnalysisError("source_version_changed")
            if shared_owned:
                shared = db.get(PublicArticleExtraction, shared_key)
                if shared.run_id != calls.run_id or shared.status != "running":
                    raise AnalysisError("extraction_lease_lost")
                shared.status, shared.facts, shared.coverage, shared.limitations = "completed", facts, coverage, limits
            event.version += 1
            event.topics = sorted({t for f in bound for t in f["attributes"]["topics"]})
            event.topic_method = "quote_and_llm_entailment"
            event.extraction_status = "extracted" if coverage["complete"] else "partially_extracted"
            source.extraction_status = event.extraction_status
            record.event_version, record.status, record.facts = event.version, "completed", bound
            record.limitations = limits
            record.coverage = {**coverage, "public_cache_reused": bool(shared_key and not shared_owned)}
    except Exception:
        with factory.begin() as db:
            record = db.get(Extraction, key)
            if record and record.run_id == calls.run_id:
                record.status = "failed"
            shared = db.get(PublicArticleExtraction, shared_key) if shared_owned else None
            if shared and shared.run_id == calls.run_id:
                shared.status = "failed"  # Unknown outcomes never release a budget reservation or retry.
        raise
    return key
