"""Scoped extraction → relevance → bank tools → verified insight/answer."""
from __future__ import annotations

import time
from pathlib import Path

from sqlalchemy import or_, select, update
from sqlalchemy.exc import IntegrityError

from app.core import feature_evidence
from app.news import Event, SourceDocument, new_id, serialize_event
from . import AGENT_ROLES, VERSION
from .agent import Tools, analyze
from .extraction import EXTRACTION_VERSION, extract
from .provider import Calls, Provider, hashed
from .settings import AnalysisError
from .storage import AnalysisLease, AnalysisRun, Answer, Extraction, Insight, LLMCall


def company_profile(company, role):
    return {"id": company.id, "role": role, "profile_version": company.profile_version,
            "industry": {"okved_code": company.okved_code, "okved_name": company.okved_name},
            "region_registration": company.region, "city_registration": company.city,
            "products": company.products, "fts_flag": company.fts_flag, "fts_countries": company.countries,
            "features": feature_evidence(company), "quality_flags": company.quality_flags,
            "limitations": ["company_snapshot_date_unknown", "revenue_unit_unknown",
                            "contract_balances_rates_and_maturities_unknown", "registration_is_not_operations"]}


def event_payload(db, event, source):
    snapshot = db.scalar(select(Extraction).where(Extraction.event_id == event.id,
                                                 Extraction.event_version == event.version,
                                                 Extraction.source_id == source.id,
                                                 Extraction.source_version == source.version,
                                                 Extraction.status == "completed"))
    payload = serialize_event(event, source)
    payload["facts"] = snapshot.facts if snapshot else []
    payload["extraction_limitations"] = snapshot.limitations if snapshot else ["facts_not_extracted"]
    payload["extraction_coverage"] = snapshot.coverage if snapshot else None
    return payload


def normalized_event(db, event, source, extraction=None):
    from alphich_kb.personalization.inputs import from_backend_event, normalize_event
    payload = event_payload(db, event, source)
    if extraction is not None:
        payload.update(version=extraction.event_version, facts=extraction.facts,
                       extraction_limitations=extraction.limitations, extraction_coverage=extraction.coverage)
    normalized = normalize_event(from_backend_event(payload, [{"id": source.id, "version": source.version,
        "company_id": source.company_id, "title": source.title, "text": source.body, "url": source.source_url,
        "published_at": source.published_at, "received_at": source.received_at}]))
    normalized["extraction_limitations"] = payload["extraction_limitations"]
    normalized["extraction_coverage"] = payload["extraction_coverage"]
    return normalized


class Runtime:
    def __init__(self, factory, settings, provider=None, bank=None):
        self.factory, self.settings = factory, settings
        self.provider = provider or Provider(factory, settings)
        self._bank = bank
        self._bank_loaded = bank is not None
        self.encoder = None

    def bank(self):
        if not self._bank_loaded:
            if not Path(self.settings.bank_index).is_dir():
                self._bank_loaded = True
                return None
            from alphich_kb.embeddings import E5Encoder, EncoderUnavailable
            from alphich_kb.retriever import BankRetriever
            try:
                self.encoder = E5Encoder(self.settings.model_dir, self.settings.model_manifest)
            except EncoderUnavailable:
                self.encoder = None
            retriever = BankRetriever(self.settings.bank_index, self.encoder)
            allowed = frozenset(retriever.documents) if self.settings.allowed_bank_ids is None else self.settings.allowed_bank_ids
            self._bank = (retriever, allowed)
            self._bank_loaded = True
        return self._bank

    def versions(self):
        bank = self.bank()
        return {"runtime": VERSION, "extraction": EXTRACTION_VERSION,
                "agent_graph": {"roles": list(AGENT_ROLES), "research_max_calls": 2, "verification_max_calls": 2,
                                "shared_max_calls": self.settings.max_calls, "shared_max_tools": self.settings.max_tools},
                "models": [self.settings.extraction_model, self.settings.analysis_model],
                "generation": {key: [m.input_limit, m.output_limit, m.reasoning_enabled, m.response_mode]
                               for key, m in self.settings.models.items()
                               if key in {self.settings.extraction_model, self.settings.analysis_model}},
                "budget_mode": self.settings.budget_mode,
                "receipt_rub_per_unit": str(self.settings.receipt_rub_per_unit) if self.settings.receipt_rub_per_unit is not None else None,
                "bank_index": bank[0].manifest["version"] if bank else None,
                "allowed_bank_ids": sorted(bank[1]) if bank else [],
                "embedding_space": bank[0].manifest.get("embedding_space_id") if bank else None,
                "bank_retrieval_mode": "dense" if bank and bank[0].encoder is not None else "lexical_fallback" if bank else "unavailable"}

    def extract_event(self, event_id, calls):
        with self.factory() as db:
            event = db.get(Event, event_id)
            source = db.get(SourceDocument, event.source_id)
            model = self.settings.models[self.settings.extraction_model]
            verifier = self.settings.models[self.settings.analysis_model]
            identity = [source.id, source.version, source.content_hash, EXTRACTION_VERSION, self.settings.extraction_model,
                        model.input_limit, model.output_limit, model.reasoning_enabled, model.response_mode,
                        verifier.id, verifier.input_limit, verifier.output_limit, verifier.reasoning_enabled, verifier.response_mode]
            key = "extraction_" + hashed(identity)
            unresolved = db.scalar(select(LLMCall.id).join(Extraction, Extraction.run_id == LLMCall.run_id).where(
                Extraction.source_id == source.id, Extraction.status == "failed", LLMCall.cost_micros.is_(None)).limit(1))
            if unresolved:
                raise AnalysisError("extraction_outcome_unknown")
            existing = db.get(Extraction, key)
            if existing and existing.status == "completed":
                return key
            if existing:
                raise AnalysisError("extraction_in_progress" if existing.status == "running" and
                                    existing.lease_expires_at > int(time.time()) else "extraction_outcome_unknown")
            snapshot = {"id": source.id, "version": source.version, "title": source.title, "text": source.body,
                        "received_at": source.received_at}
        try:
            with self.factory.begin() as db:
                db.add(Extraction(id=key, source_id=snapshot["id"], source_version=snapshot["version"],
                                  event_id=event_id, event_version=0, status="running", facts=[], limitations=[],
                                  coverage={}, run_id=calls.run_id, lease_expires_at=int(time.time()) + 600))
        except IntegrityError:
            raise AnalysisError("extraction_in_progress") from None
        try:
            facts, coverage, limits = extract(snapshot, calls)
        except Exception:
            with self.factory.begin() as db:
                failed = db.get(Extraction, key)
                if failed.run_id == calls.run_id:
                    failed.status = "failed"
            raise
        calls.heartbeat()
        with self.factory.begin() as db:
            event = db.scalar(select(Event).where(Event.id == event_id).with_for_update())
            source = db.scalar(select(SourceDocument).where(SourceDocument.id == snapshot["id"]).with_for_update())
            record = db.get(Extraction, key)
            if record.run_id != calls.run_id or record.status != "running":
                raise AnalysisError("extraction_lease_lost")
            if source.version != snapshot["version"] or source.body != snapshot["text"]:
                raise AnalysisError("source_version_changed")
            event.version += 1
            event.topics = sorted({topic for f in facts for topic in f["attributes"]["topics"]})
            event.topic_method = "quote_and_llm_entailment"
            event.extraction_status = "extracted" if coverage["complete"] else "partially_extracted"
            source.extraction_status = event.extraction_status
            record.event_version, record.status, record.facts = event.version, "completed", facts
            record.limitations, record.coverage = limits, coverage
        return key

    def process(self, task, job, heartbeat):
        from alphich_kb.personalization.inputs import SelectionScope
        from alphich_kb.personalization.selector import NewsSelector
        self.settings.require_ready()
        run_id = "run_" + hashed(job.id)[:32]
        with self.factory.begin() as db:
            run = db.get(AnalysisRun, run_id)
            if run is None:
                db.add(AnalysisRun(id=run_id, company_id=job.company_id, role=task.role, as_of=task.as_of,
                                   status="running", steps=[], versions=self.versions(), created_at=int(time.time())))
            else:
                record = db.scalar(select(Answer).where(Answer.run_id == run_id)) if task.parent_insight_id else db.scalar(
                    select(Insight).where(Insight.run_id == run_id))
                if record:
                    run.status, run.verification = "recovered", record.payload["verification"]
                    return ("answer" if task.parent_insight_id else "insight"), record.id
        lease_key = None

        def lease_heartbeat():
            heartbeat()
            with self.factory.begin() as db:
                if tools is not None:
                    db.get(AnalysisRun, run_id).steps = list(tools.steps)
                if lease_key is not None:
                    changed = db.execute(update(AnalysisLease).where(AnalysisLease.cache_key == lease_key,
                        AnalysisLease.run_id == run_id).values(expires_at=int(time.time()) + 600))
                    if changed.rowcount != 1:
                        raise AnalysisError("analysis_cache_lease_lost")

        calls = Calls(self.provider, run_id, lease_heartbeat)
        tools = None
        try:
            scope = SelectionScope(job.company_id, task.as_of, frozenset({task.event_id}), task.role)
            if task.parent_insight_id:
                with self.factory() as db:
                    parent = db.get(Insight, task.parent_insight_id)
                    if parent is None or parent.company_id != job.company_id or parent.role != task.role:
                        raise AnalysisError("parent_insight_unavailable")
                    # Enqueue-time authorization can become stale before this
                    # worker runs. Saved bank evidence must pass the current
                    # policy before entering tools or any model prompt.
                    if self.settings.allowed_bank_ids is not None and any(
                            evidence.get("document_id") and evidence["document_id"] not in self.settings.allowed_bank_ids
                            for evidence in parent.evidence.values()):
                        raise AnalysisError("evidence_outside_scope")
                    snapshot, registry = parent.snapshot, parent.evidence
                event, selection, related = snapshot["event"], snapshot["selection"], snapshot["related_events"]
                bank = self.bank() if snapshot["versions"] == self.versions() else None
                tool_scope = SelectionScope(job.company_id, task.as_of, frozenset([event["id"], *[e["id"] for e in related]]), task.role)
                tools = Tools(task.profile, event, selection, tool_scope, bank, related)
                tools.registry.update(registry)
                if not event["facts"]:
                    raise AnalysisError("insufficient_news_facts")
            else:
                with self.factory() as db:
                    event = db.get(Event, task.event_id)
                    source = db.get(SourceDocument, event.source_id) if event else None
                    if not event or not source or event.company_id != job.company_id or source.company_id != job.company_id:
                        raise AnalysisError("event_outside_scope")
                    before = normalized_event(db, event, source)
                    if scope.excludes(before):
                        raise AnalysisError("event_unavailable_as_of")
                extraction_id = self.extract_event(task.event_id, calls)
                with self.factory() as db:
                    event = normalized_event(db, db.get(Event, task.event_id), db.get(SourceDocument, source.id), db.get(Extraction, extraction_id))
                    related = [normalized_event(db, e, s) for e, s in db.execute(select(Event, SourceDocument).join(
                        SourceDocument, Event.source_id == SourceDocument.id).where(
                        Event.company_id == job.company_id, SourceDocument.company_id == job.company_id,
                        SourceDocument.received_at <= scope.as_of,
                        or_(SourceDocument.published_at.is_(None), SourceDocument.published_at <= scope.as_of))
                        .order_by(SourceDocument.received_at.desc(), Event.id.desc()).limit(60)).all()]
                selected = NewsSelector([event], self.encoder).select(task.profile, scope)
                selection = next((item for item in selected["items"] if item["event_id"] == task.event_id), {})
                tool_scope = SelectionScope(job.company_id, task.as_of, frozenset([event["id"], *[e["id"] for e in related]]), task.role)
                tools = Tools(task.profile, event, selection, tool_scope, self.bank(), related)
                snapshot = {"profile": task.profile, "event": event, "selection": selection,
                            "related_events": related, "versions": self.versions()}
                registry = tools.registry
            related_version = sorted((e["id"], e["version"], e["snapshot_hash"]) for e in related if not tools.scope.excludes(e))
            cache_key = hashed([task.profile, event["snapshot_hash"], related_version, task.role, scope.as_of, self.versions()])
            if not task.parent_insight_id:
                with self.factory() as db:
                    cached = db.scalar(select(Insight).where(Insight.cache_key == cache_key))
                    if cached:
                        self.save_run(run_id, "cache_hit", [{"action": "cache_hit", "insight_id": cached.id}], {"status": "cached"})
                        return "insight", cached.id
                with self.factory() as db:
                    lease = db.get(AnalysisLease, cache_key)
                    if lease and lease.run_id != run_id:
                        raise AnalysisError("analysis_in_progress" if lease.expires_at > int(time.time()) else "analysis_outcome_unknown")
                if lease is None:
                    try:
                        with self.factory.begin() as db:
                            db.add(AnalysisLease(cache_key=cache_key, run_id=run_id, expires_at=int(time.time()) + 600))
                    except IntegrityError:
                        raise AnalysisError("analysis_in_progress") from None
                lease_key = cache_key
            if not event["facts"] or (not task.parent_insight_id and selection.get("status") != "applicable_with_conditions"):
                draft = {"claims": [{"id": f"c{i+1}", "kind": "news_fact", "text": f["text"],
                                     "evidence_ids": [f["id"]], "conditions": []} for i, f in enumerate(event["facts"][:5])],
                         "risks": [], "opportunities": [], "next_steps": [],
                         "missing_data": ["Подтверждённых оснований для персонального вывода недостаточно"]}
                verification = {"status": "insufficient_evidence", "issues": ["no_verified_applicability_or_facts"]}
            else:
                draft, verification = analyze(tools, calls, task.question)
            statuses = [c["bank_view_status"] for c in tools.bank_contexts]
            bank_status = next((s for s in ("conflicting", "found", "historical_only") if s in statuses), "not_found")
            bank_claim_ids = [c["id"] for c in draft["claims"] if c["kind"] == "bank_view"]
            if not bank_claim_ids:
                bank_status = "not_found"
            elif bank_status == "not_found":
                # A frozen question may retain authorized reviewed evidence
                # even when the current index/version is unavailable for search.
                bank_status = "found"
            payload = {"company_id": job.company_id, "role": task.role, "event_id": event["id"],
                       "event_version": event["version"], "profile_version": task.profile["profile_version"],
                       "as_of": scope.as_of, "analysis_mode": "as_of", "status": "insufficient_evidence" if verification["status"] == "insufficient_evidence" else "ready_with_conditions",
                       "headline": "Разбор: " + event["title"], "relevance_reasons": selection.get("reasons", []),
                       **draft, "bank_view": {"status": bank_status,
                           "claim_ids": bank_claim_ids,
                           "reason": None if bank_claim_ids else "No verified applicable bank claim included in this answer",
                           "scope": "loaded_authorized_corpus_only"},
                       "verification": verification, "run_id": run_id,
                       "extraction_coverage": event.get("extraction_coverage"),
                       "limitations": sorted(set(selection.get("limitations", []) + event.get("extraction_limitations", []) + ["llm_semantic_check_is_not_human_review",
                           "source_truth_not_independently_confirmed", "company_snapshot_date_unknown"] +
                           [l for c in tools.bank_contexts for l in c.get("limitations", [])]))}
            lease_heartbeat()
            with self.factory.begin() as db:
                if task.parent_insight_id:
                    result_id = new_id("answer")
                    payload.update(id=result_id, insight_id=task.parent_insight_id, question=task.question)
                    db.add(Answer(id=result_id, insight_id=task.parent_insight_id, company_id=job.company_id, role=task.role,
                                  payload=payload, evidence=tools.registry, run_id=run_id, created_at=int(time.time())))
                    result_type = "answer"
                else:
                    result_id, result_type = new_id("insight"), "insight"
                    payload["id"] = result_id
                    db.add(Insight(id=result_id, company_id=job.company_id, role=task.role, event_id=event["id"],
                                   cache_key=cache_key, payload=payload, evidence=tools.registry, snapshot=snapshot,
                                   run_id=run_id, created_at=int(time.time())))
            self.save_run(run_id, "succeeded", tools.steps, verification)
            return result_type, result_id
        except Exception as error:
            self.save_run(run_id, "failed", tools.steps if tools else [],
                          {"status": "failed", "issues": [error.code if isinstance(error, AnalysisError) else "analysis_failed"]})
            raise

    def save_run(self, run_id, status, steps, verification):
        with self.factory.begin() as db:
            run = db.get(AnalysisRun, run_id)
            run.status, run.steps, run.verification = status, steps, verification
