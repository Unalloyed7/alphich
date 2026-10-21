"""Bounded local acceptance on actual stored article snapshots, optional live LLM.

Uses an isolated new SQLite database, the real API, ingestion worker, automatic
dispatcher, analysis worker, E5 and bank index. Never resets an existing ledger.
"""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import json
import os
from urllib.parse import urlsplit
from uuid import uuid4
from xml.sax.saxutils import escape

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.core import Base, Company, make_session_factory
from app.news import Event, Job, SourceDocument
from app.rss import FeedConfig, collect_once
from app.rss_fetch import Download
from app.worker import process_next_job
from .automation import AutomationConfig, AutomationRecord, dispatch_once
from .news_index import encoder_for, sync_index
from .offline import save_local
from .provider import hashed
from .service import Runtime
from .settings import AnalysisError, micros
from .storage import Answer, Insight, LLMCall
from .worker import process_next


class SnapshotReplay:
    """Replay frozen, explicitly labelled source text; no fabricated model facts.

    The input corpus records actual download provenance. This verifies local
    ingestion and AI, while source availability is checked by the live collector.
    """
    def __init__(self, events):
        self.events = events
        self.articles = {row["sources"][0]["url"]: row for row in events}
        host = urlsplit(next(iter(self.articles))).hostname
        self.url = "https://" + host + "/alphich-frozen-acceptance.xml"

    def get(self, url, hosts, validators=None):
        if url == self.url:
            from email.utils import format_datetime
            items = []
            for row in self.events:
                published = row.get("published_at") or row["sources"][0].get("published_at")
                stamp = format_datetime(datetime.fromisoformat(published.replace("Z", "+00:00"))) if published else ""
                items.append("<item><title>" + escape(row["title"]) + "</title><link>" + escape(row["sources"][0]["url"])
                    + "</link><pubDate>" + stamp + "</pubDate></item>")
            return Download(url, 200, {"content-type": "application/rss+xml"},
                ("<rss><channel>" + "".join(items) + "</channel></rss>").encode())
        row = self.articles[url]
        # Faithfully replay text through the HTML extractor. Labelled as replay
        # in the report; no claim that an external RSS feed has this URL.
        body = "<html><head><meta charset='utf-8'></head><body><main><article>" + "".join(
            "<p>" + escape(p) + "</p>" for p in row["sources"][0]["text"].splitlines() if p.strip()) + "</article></main></body></html>"
        return Download(url, 200, {"content-type": "text/html; charset=utf-8"}, body.encode())


def _rows(path, field):
    data = json.loads(Path(path).read_text())
    return data[field] if isinstance(data, dict) else data


def run_acceptance(settings, profiles_path, events_path, output, *, live=False, max_articles=3,
                   max_cards=2, budget_rub="2", with_question=True, runtime_factory=Runtime):
    from app.api import create_app
    from alphich_kb.personalization.profiles import PRODUCTS
    if not 1 <= max_articles <= 12 or not 1 <= max_cards <= 6 or not 0 < micros(budget_rub) <= 10_000_000:
        raise ValueError("invalid_acceptance_limits")
    root = Path("data/processed").resolve()
    output = Path(output).resolve()
    if not output.is_relative_to(root):
        raise ValueError("acceptance_output_must_be_ignored")
    profiles, events = _rows(profiles_path, "companies"), _rows(events_path, "events")[:max_articles]
    if not profiles or len(profiles) > 5 or not events:
        raise ValueError("bounded_profiles_and_news_required")
    companies = [row.get("company_id", row.get("id")) for row in profiles]
    if len(set(companies)) != len(companies):
        raise ValueError("duplicate_acceptance_company")
    run_id = "acceptance_" + uuid4().hex[:16]
    database = output.with_name(run_id + ".db")
    if database.exists():
        raise ValueError("acceptance_database_must_be_new")
    report = {"version": "local-ai-acceptance-v1", "run_id": run_id,
        "started_at": datetime.now(timezone.utc).isoformat(), "live": live,
        "replayed_article_snapshots": True, "source_network_test_separate": True,
        "profiles_hash": hashed(profiles), "events_hash": hashed(events), "database": str(database),
        "max_articles": max_articles, "max_cards": max_cards, "budget_limit_rub": str(budget_rub),
        "semantics_independently_reviewed": False, "phases": [], "errors": []}
    output.parent.mkdir(parents=True, exist_ok=True)
    engine, factory = make_session_factory("sqlite:///" + str(database))
    Base.metadata.create_all(engine)
    database.chmod(0o600)
    old_allowed = os.environ.get("DEMO_COMPANY_IDS")
    os.environ["DEMO_COMPANY_IDS"] = ",".join(companies)
    configured = replace(settings, daily_limit_micros=min(settings.daily_limit_micros, micros(budget_rub)))
    def persist():
        with factory() as db:
            calls = list(db.scalars(select(LLMCall)))
            report["calls"] = [{"id": c.id, "model": c.model, "purpose": c.purpose, "status": c.status,
                "cost_micros": c.cost_micros, "input_tokens": c.input_tokens, "output_tokens": c.output_tokens,
                "reserved_micros": c.reserved_micros, "error": c.error} for c in calls]
            report["cost_known_micros"] = sum(c.cost_micros or 0 for c in calls)
            report["unknown_reserved_micros"] = sum(c.reserved_micros for c in calls if c.cost_micros is None)
            report["generative_calls"] = len(calls)
        save_local(output, report)
    try:
        with factory.begin() as db:
            for row in profiles:
                p = row.get("profile", row)
                industry, fts = p.get("industry", {}), p.get("fts", {})
                db.add(Company(id=row.get("company_id", p.get("id")), client_type=p.get("client_type"),
                    okved_code=industry.get("code", industry.get("okved_code")),
                    okved_name=industry.get("name", industry.get("okved_name")),
                    region=p.get("registration_region", p.get("region_registration")),
                    city=p.get("registration_city", p.get("city_registration")), revenue_raw=p.get("revenue_raw"),
                    products={column: p.get("products", {}).get(column, p.get("products", {}).get(alias))
                              for alias, column in PRODUCTS.items()}, fts_flag=fts.get("flag", p.get("fts_flag")),
                    countries=fts.get("countries", p.get("fts_countries", [])),
                    profile_version=row["profile_version"], quality_flags=p.get("quality_flags", []),
                    source_rows=[p["source_record"]] if type(p.get("source_record")) is int else []))
        replay = SnapshotReplay(events)
        hosts = sorted({urlsplit(url).hostname for url in replay.articles})
        feed = FeedConfig(id="acceptance", url=replay.url, article_hosts=hosts, max_items=max_articles)
        collection = collect_once(factory, [feed], companies, fetcher=replay, force=True)[0]
        if collection["errors"] or collection["queued"] != len(events) * len(companies):
            raise AnalysisError("acceptance_ingestion_failed")
        while process_next_job(factory):
            pass
        report["phases"].append({**collection, "phase": "ingestion", "status": "passed"})
        index = encoder_for(configured)
        index_results = []
        for _ in range(len(events) * len(companies)):
            batch = sync_index(factory, index, companies, limit=1)
            index_results.append(batch)
            if not batch["indexed"]:
                break
        report["phases"].append({"phase": "persistent_e5_index", "status": "passed" if index else "unavailable",
            "indexed_sources": sum(r["indexed"] for r in index_results), "space_id": index.space_id if index else None})
        runtime = runtime_factory(factory, configured)
        report["runtime_versions"] = runtime.versions()
        if not live:
            report["status"] = "preflight_only_no_generation"
            persist()
            return report
        configured.require_ready()
        config = AutomationConfig(enabled=True, company_ids=tuple(companies), roles=("cfo",),
            extraction_limit=min(100, len(events) * len(companies)), card_limit=max_cards)
        for tick in range(4):
            dispatch = dispatch_once(factory, configured, config)
            report["phases"].append({"phase": "dispatch", "tick": tick, **dispatch})
            persist()
            for _ in range(len(events)*len(companies) + max_cards):
                if not process_next(factory, runtime):
                    break
                persist()
        with factory() as db:
            records = list(db.scalars(select(AutomationRecord)))
            report["decisions"] = [{"company_id": r.company_id, "event_id": r.event_id,
                "source_url": db.get(SourceDocument, r.source_id).source_url, "status": r.status,
                "grade": r.grade, "assessment": r.assessment, "error": r.error} for r in records]
            cards = list(db.scalars(select(Insight)))
            report["cards"] = [c.payload for c in cards]
            report["bank_evidence_count"] = sum(e["kind"] == "bank_claim" for c in cards for e in c.evidence.values())
        with TestClient(create_app(factory, engine)) as client:
            for company_id in companies:
                token = client.post("/api/v1/demo-session", json={"company_id": company_id, "role": "cfo"}).json()["token"]
                headers = {"X-Demo-Session": token}
                result = client.get(f"/api/v1/companies/{company_id}/feed", headers=headers)
                if result.status_code != 200 or any(item["relevance_grade"] < 4 for item in result.json()["items"]):
                    raise AnalysisError("acceptance_feed_failed")
            if cards and with_question:
                parent = cards[0]
                token = client.post("/api/v1/demo-session", json={"company_id": parent.company_id, "role": parent.role}).json()["token"]
                headers = {"X-Demo-Session": token, "Idempotency-Key": run_id + ":question"}
                # Route configuration uses the same bounded config, never the caller's unbounded .env.
                saved_cap = os.environ.get("LLM_DAILY_LIMIT_RUB")
                os.environ["LLM_DAILY_LIMIT_RUB"] = str(budget_rub)
                try:
                    queued = client.post(f"/api/v1/insights/{parent.id}/questions", headers=headers,
                        json={"question": "Какие данные компании нужно уточнить перед решением по этой новости?"})
                    if queued.status_code != 202:
                        raise AnalysisError("acceptance_question_enqueue_failed")
                    process_next(factory, runtime)
                    with factory() as db:
                        job = db.get(Job, queued.json()["id"])
                        report["question_status"] = job.status
                        if job.status == "succeeded":
                            report["answer"] = db.get(Answer, job.result_id).payload
                        else:
                            report["errors"].append(job.error)
                finally:
                    if saved_cap is None: os.environ.pop("LLM_DAILY_LIMIT_RUB", None)
                    else: os.environ["LLM_DAILY_LIMIT_RUB"] = saved_cap
        report["status"] = "passed" if cards and not report["errors"] and all(r.grade is not None for r in records) else "needs_review"
        persist()
        return report
    except Exception as exc:
        report["status"] = "failed"
        report["errors"].append(exc.code if isinstance(exc, AnalysisError) else type(exc).__name__)
        persist()
        raise
    finally:
        if old_allowed is None: os.environ.pop("DEMO_COMPANY_IDS", None)
        else: os.environ["DEMO_COMPANY_IDS"] = old_allowed
        engine.dispose()


def resume_question(settings, previous_report, output, question, *, runtime_factory=Runtime):
    """Append a distinct focused question within the ORIGINAL ledger and cap.

    The previous failed result stays intact. Repeating this exact operation is
    idempotent and never bypasses an uncertain request or replenishes reserves.
    """
    from app.api import create_app
    root = Path("data/processed").resolve()
    previous_report, output = Path(previous_report).resolve(), Path(output).resolve()
    if not previous_report.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError("acceptance_paths_must_be_ignored")
    before = json.loads(previous_report.read_text())
    database = Path(before["database"]).resolve()
    if not database.is_relative_to(root) or not database.is_file():
        raise ValueError("existing_acceptance_ledger_required")
    configured = replace(settings, daily_limit_micros=min(settings.daily_limit_micros, micros(before["budget_limit_rub"])))
    engine, factory = make_session_factory("sqlite:///" + str(database))
    old_allowed = os.environ.get("DEMO_COMPANY_IDS")
    try:
        with factory() as db:
            card = db.scalar(select(Insight).order_by(Insight.created_at, Insight.id))
            if card is None: raise ValueError("verified_parent_card_required")
            os.environ["DEMO_COMPANY_IDS"] = card.company_id
            company_id, role, card_id = card.company_id, card.role, card.id
        runtime = runtime_factory(factory, configured)
        resume_key = "resume:" + hashed([card_id, question])
        from .storage import AnalysisTask
        with factory() as db:
            previous_job_id = db.scalar(select(Job.id).join(AnalysisTask, AnalysisTask.job_id == Job.id).where(
                Job.company_id == company_id, Job.operation == "insight_question", Job.idempotency_key == resume_key,
                AnalysisTask.parent_insight_id == card_id, AnalysisTask.role == role))
        with TestClient(create_app(factory, engine)) as client:
            token = client.post("/api/v1/demo-session", json={"company_id": company_id, "role": role}).json()["token"]
            headers = {"X-Demo-Session": token, "Idempotency-Key": resume_key}
            if previous_job_id:
                question_job_id = previous_job_id
            else:
                queued = client.post(f"/api/v1/insights/{card_id}/questions", headers=headers, json={"question": question})
                if queued.status_code != 202: raise AnalysisError("acceptance_question_enqueue_failed")
                question_job_id = queued.json()["id"]
            for _ in range(4):
                with factory() as db:
                    status = db.get(Job, question_job_id).status
                if status not in {"queued", "running"}: break
                if not process_next(factory, runtime): break
            with factory() as db:
                job = db.get(Job, question_job_id)
                calls = list(db.scalars(select(LLMCall)))
                result = {"version": "local-ai-followup-v1", "base_report": str(previous_report),
                    "database": str(database), "budget_limit_rub": before["budget_limit_rub"],
                    "status": "passed" if job.status == "succeeded" else "failed", "error": job.error,
                    "question_status": job.status, "parent_insight_id": card_id, "job_id": job.id,
                    "generative_calls_total": len(calls), "cost_known_micros_total": sum(c.cost_micros or 0 for c in calls),
                    "unknown_reserved_micros": sum(c.reserved_micros for c in calls if c.cost_micros is None),
                    "answer": db.get(Answer, job.result_id).payload if job.status == "succeeded" else None}
                if result["answer"]:
                    for claim in result["answer"]["claims"]:
                        for evidence_id in claim["evidence_ids"]:
                            evidence = client.get(f"/api/v1/evidence/{evidence_id}", headers=headers,
                                params={"insight_id": card_id, "answer_id": job.result_id})
                            if evidence.status_code != 200: raise AnalysisError("acceptance_evidence_failed")
        save_local(output, result)
        return result
    finally:
        if old_allowed is None: os.environ.pop("DEMO_COMPANY_IDS", None)
        else: os.environ["DEMO_COMPANY_IDS"] = old_allowed
        engine.dispose()


def main():
    from .gateway import load_env_file
    from .settings import Settings
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file")
    parser.add_argument("--profiles")
    parser.add_argument("--events")
    parser.add_argument("--resume-report")
    parser.add_argument("--question", default="Какие сведения о товарном ассортименте компании нужно уточнить по этой новости?")
    parser.add_argument("--output", default="data/processed/ai-acceptance.local.json")
    parser.add_argument("--max-articles", type=int, default=3)
    parser.add_argument("--max-cards", type=int, default=2)
    parser.add_argument("--budget-rub", default="2")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    if args.env_file: load_env_file(args.env_file)
    try:
        if args.resume_report:
            if not args.live: parser.error("resume-report requires explicit --live; the original ledger cap remains in force")
            result = resume_question(Settings.from_env(), args.resume_report, args.output, args.question)
            print(json.dumps({k: result.get(k) for k in ("status", "question_status", "cost_known_micros_total",
                "generative_calls_total", "unknown_reserved_micros", "error")}))
            return 0 if result["status"] == "passed" else 2
        if not args.profiles or not args.events: parser.error("profiles and events are required")
        result = run_acceptance(Settings.from_env(), args.profiles, args.events, args.output,
            live=args.live, max_articles=args.max_articles, max_cards=args.max_cards, budget_rub=args.budget_rub)
        print(json.dumps({k: result.get(k) for k in ("status", "run_id", "generative_calls", "cost_known_micros",
            "unknown_reserved_micros", "question_status", "errors")}, ensure_ascii=False))
        return 0 if result["status"] in {"passed", "preflight_only_no_generation"} else 2
    except (AnalysisError, ValueError):
        print(json.dumps({"status": "failed", "details": str(args.output)}))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
