"""Collector-proven sources -> real queue -> shared facts -> grades -> scoped cards."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone
import time

import pytest
from sqlalchemy import func, select

from app.core import Company
from app.news import Event, Job, SourceDocument
from app.rss import RssArticle
from app.intelligence.automation import AutomationConfig, AutomationQuota, AutomationRecord, dispatch_once
from app.intelligence.storage import Insight, LLMCall
from app.intelligence.worker import process_next
from test_intelligence import store, runtime, ScriptedGateway, stored_event
from test_ai_offline_features import public_copies


def copies(store):
    ids = public_copies(store)
    _, factory, _, _ = store
    stamp = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with factory.begin() as db:
        article = db.scalar(select(RssArticle))
        article.published_at = article.fetched_at = stamp
        for source in db.scalars(select(SourceDocument)):
            source.published_at = source.received_at = source.fetched_at = stamp
    return ids


def config(**changes):
    return AutomationConfig(enabled=True, company_ids=("DEMO_A", "DEMO_B"), **changes)


def drain(factory, engine):
    count = 0
    while process_next(factory, engine):
        count += 1
        assert count < 20
    return count


def test_two_companies_share_extraction_and_only_grade_4_gets_a_personal_card(store):
    client, factory, headers, settings = store
    copies(store)
    engine, gateway = runtime(factory, settings)
    first = dispatch_once(factory, settings, config())
    assert first["extractions_queued"] == 2 and first["cards_queued"] == 0
    assert drain(factory, engine) == 2
    assert [name for name, _ in gateway.seen] == ["ExtractionDraft", "ExtractionReview"]
    second = dispatch_once(factory, settings, config())
    assert second["cards_queued"] == 1 and second["below_threshold"] == 1
    assert drain(factory, engine) == 1
    dispatch_once(factory, settings, config())
    with factory() as db:
        cards = list(db.scalars(select(Insight)))
        assert len(cards) == 1 and cards[0].company_id == "DEMO_A" and cards[0].payload["relevance_grade"] == 4
        assert db.scalar(select(func.count()).select_from(Job).where(Job.operation == "automatic_insight")) == 1
        assert {r.status for r in db.scalars(select(AutomationRecord))} == {"completed", "below_threshold"}
    feed = client.get("/api/v1/companies/DEMO_A/feed", headers=headers["DEMO_A"]).json()
    assert feed["items"][0]["insight"]["id"] == cards[0].id
    assert client.get("/api/v1/insights/" + cards[0].id, headers=headers["DEMO_B"]).status_code == 404
    assert client.get("/api/v1/companies/DEMO_A/automation", headers=headers["DEMO_B"]).status_code == 403
    assert client.get("/api/v1/companies/DEMO_A/automation", headers=headers["DEMO_A"]).json()["items"][0]["grade"] == 4
    before = len(gateway.seen)
    for _ in range(3):
        report = dispatch_once(factory, settings, config())
        assert report["extractions_queued"] == report["cards_queued"] == 0
        assert not process_next(factory, engine)
    assert len(gateway.seen) == before


def test_disabled_or_unconfigured_automation_creates_no_paid_work(store):
    _, factory, _, settings = store
    copies(store)
    assert dispatch_once(factory, settings, replace(config(), enabled=False))["enabled"] is False
    assert dispatch_once(factory, replace(settings, api_key=""), config())["errors"] == ["llm_not_configured"]
    with factory() as db:
        assert not list(db.scalars(select(AutomationRecord)))
        assert not list(db.scalars(select(LLMCall)))


def test_only_collector_proven_and_dated_sources_can_start_automatic_models(store):
    client, factory, headers, settings = store
    stored_event(client, factory, headers["DEMO_A"])
    with factory.begin() as db:
        source = db.scalar(select(SourceDocument))
        source.kind = "rss_article"  # A client-provided marker is insufficient.
        source.published_at = source.received_at = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    assert dispatch_once(factory, settings, config())["extractions_queued"] == 0
    with factory() as db:
        assert not list(db.scalars(select(AutomationRecord)))


@pytest.mark.parametrize("published", [None, "2099-01-01T00:00:00+00:00", "2020-01-01T00:00:00+00:00"])
def test_missing_future_or_ancient_publication_is_not_automatically_billed(store, published):
    _, factory, _, settings = store
    copies(store)
    with factory.begin() as db:
        db.scalar(select(RssArticle)).published_at = published
        for source in db.scalars(select(SourceDocument)):
            source.published_at = published
    assert dispatch_once(factory, settings, config())["extractions_queued"] == 0


def test_daily_extraction_quota_is_durable_and_repeated_ticks_do_not_spend_it_twice(store):
    _, factory, _, settings = store
    copies(store)
    cfg = config(extraction_limit=1)
    assert dispatch_once(factory, settings, cfg)["extractions_queued"] == 1
    assert dispatch_once(factory, settings, cfg)["extractions_queued"] == 0
    with factory() as db:
        assert db.scalar(select(AutomationQuota.extraction_jobs)) == 1
    next_day = int(time.time()) + 86400
    assert dispatch_once(factory, settings, cfg, now=next_day)["extractions_queued"] == 1
    with factory() as db:
        assert sorted(db.scalars(select(AutomationQuota.extraction_jobs))) == [1, 1]


def test_both_roles_reuse_one_fact_job_per_company_and_keep_separate_cards(store):
    _, factory, _, settings = store
    copies(store)
    cfg = config(roles=("cfo", "ceo"))
    engine, gateway = runtime(factory, settings)
    assert dispatch_once(factory, settings, cfg)["extractions_queued"] == 2
    drain(factory, engine)
    assert dispatch_once(factory, settings, cfg)["cards_queued"] == 2
    drain(factory, engine)
    with factory() as db:
        assert set(db.scalars(select(Insight.role))) == {"cfo", "ceo"}
        assert db.scalar(select(func.count()).select_from(Insight)) == 2
    assert sum(name == "ExtractionDraft" for name, _ in gateway.seen) == 1


def test_uncertain_paid_outcome_stays_blocked_across_dispatch_and_restart(store):
    _, factory, _, settings = store
    copies(store)
    engine, gateway = runtime(factory, settings, ScriptedGateway(timeout=True))
    dispatch_once(factory, settings, config())
    drain(factory, engine)
    dispatch_once(factory, settings, config())
    dispatch_once(factory, settings, config())
    with factory() as db:
        assert {r.status for r in db.scalars(select(AutomationRecord))} == {"blocked"}
        assert db.scalar(select(func.count()).select_from(LLMCall)) == 1
        assert db.scalar(select(LLMCall.cost_micros)) is None
    assert len(gateway.seen) == 1 and not process_next(factory, engine)


def test_company_allowlist_and_invalid_limits_fail_before_any_dispatch(store):
    _, factory, _, settings = store
    for cfg in [replace(config(), company_ids=("OTHER",)), replace(config(), card_limit=0),
                replace(config(), roles=("administrator",)), replace(config(), roles=("cfo", "cfo"))]:
        with pytest.raises(ValueError):
            dispatch_once(factory, settings, cfg)


def test_profile_revision_makes_new_decision_without_reextracting_public_facts(store):
    _, factory, _, settings = store
    copies(store)
    engine, gateway = runtime(factory, settings)
    dispatch_once(factory, settings, config()); drain(factory, engine)
    dispatch_once(factory, settings, config()); drain(factory, engine)
    with factory.begin() as db:
        db.get(Company, "DEMO_A").profile_version = "new-profile"
    report = dispatch_once(factory, settings, config())
    assert report["extractions_queued"] == 0 and report["cards_queued"] == 1
    drain(factory, engine)
    assert sum(name == "ExtractionDraft" for name, _ in gateway.seen) == 1


def test_housing_mortgage_news_is_not_lost_by_the_free_industry_gate(store):
    _, factory, _, settings = store
    copies(store)
    body = "В России увеличился спрос на ипотеку. Банки сообщили о крупных объёмах финансирования покупки жилья по всей стране."
    with factory.begin() as db:
        article=db.scalar(select(RssArticle));article.body=body
        for source in db.scalars(select(SourceDocument)):source.body=body
    result=dispatch_once(factory,settings,config())
    assert result["extractions_queued"]==1
    with factory() as db:
        assert db.scalar(select(Job.company_id).where(Job.operation=="news_fact_extraction"))=="DEMO_A"
