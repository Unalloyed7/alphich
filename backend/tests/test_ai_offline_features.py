"""Real DB/selector and scripted extraction only; no model network access."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from sqlalchemy import select
from app.news import Event, SourceDocument
from app.rss import RssArticle, RssDelivery
from app.intelligence.extraction import extract
from app.intelligence.provider import Calls
from app.intelligence.schemas import ExtractionDraft
from app.intelligence.settings import AnalysisError
from app.intelligence.storage import Extraction, PublicArticleExtraction
from app.intelligence.windows import source_windows
from test_intelligence import BODY, runtime, store, stored_event
from alphich.data import prepare_data
from alphich_kb.common import KnowledgeError
from alphich_kb.personalization.payments import attach_payment_context
from alphich_kb.personalization.profiles import normalize_company


def public_copies(store):
    client, factory, headers, settings = store
    ids = [stored_event(client, factory, headers[c]) for c in ("DEMO_A", "DEMO_B")]
    with factory.begin() as db:
        sources = [db.get(SourceDocument, db.get(Event, e).source_id) for e in ids]
        db.add(RssArticle(id="a"*64, first_feed_id="fixture", url="https://example.org/a",
            final_url="https://example.org/a", title=sources[0].title, body=sources[0].body,
            content_hash=sources[0].content_hash, fetched_at=sources[0].received_at, published_at=None))
        for source in sources:
            source.kind, source.source_url, source.fetched_at = "rss_article", "https://example.org/a", sources[0].received_at
            db.add(RssDelivery(article_id="a"*64, company_id=source.company_id, source_id=source.id, job_id="fixture"))
    return ids


def test_public_article_has_one_extraction_pair_and_tenant_owned_evidence(store):
    _, factory, _, settings = store
    ids = public_copies(store)
    engine, gateway = runtime(factory, settings)
    for i, ident in enumerate(ids):
        engine.extract_event(ident, Calls(engine.provider, f"cache-run-{i}", lambda:None))
    assert [s for s,_ in gateway.seen] == ["ExtractionDraft", "ExtractionReview"]
    with factory() as db:
        rows = list(db.scalars(select(Extraction)))
        assert len(rows) == 2 and all(r.status == "completed" for r in rows)
        assert rows[0].facts[0]["id"] != rows[1].facts[0]["id"]
        assert all(r.facts[0]["source_id"] == r.source_id for r in rows)
        assert len(list(db.scalars(select(PublicArticleExtraction)))) == 1


def test_private_pasted_copies_never_share_fact_cache(store):
    client, factory, headers, settings = store
    ids = [stored_event(client, factory, headers[c]) for c in ("DEMO_A", "DEMO_B")]
    engine, gateway = runtime(factory, settings)
    for i, ident in enumerate(ids):
        engine.extract_event(ident, Calls(engine.provider, f"private-run-{i}", lambda:None))
    assert sum(s == "ExtractionDraft" for s,_ in gateway.seen) == 2
    with factory() as db:
        assert list(db.scalars(select(PublicArticleExtraction))) == []


def test_shared_unknown_outcome_cannot_trigger_second_company_request(store):
    from test_intelligence import ScriptedGateway
    _, factory, _, settings = store
    ids = public_copies(store)
    engine, gateway = runtime(factory, settings, ScriptedGateway(timeout=True))
    with pytest.raises(AnalysisError):
        engine.extract_event(ids[0], Calls(engine.provider, "unknown-public", lambda:None))
    with pytest.raises(AnalysisError, match="extraction_outcome_unknown"):
        engine.extract_event(ids[1], Calls(engine.provider, "second-public", lambda:None))
    assert len(gateway.seen) == 1


def test_source_windows_find_late_fact_and_keep_original_coordinates():
    quote = "Экспортная пошлина на зерно увеличена на 12%."
    body = "Описание. "*1100 + quote + " Последние подробности. "*100
    view, coverage = source_windows(body, "Экспортная пошлина на зерно")
    assert quote in view and len(view.encode()) <= 6500 and not coverage["complete"]
    class Script:
        count = 0
        provider = SimpleNamespace(settings=SimpleNamespace(extraction_model="test",analysis_model="test"))
        def ask(self, purpose, schema, system, data, **kwargs):
            self.count += 1
            if schema is ExtractionDraft:
                return schema.model_validate({"facts":[{"quote":quote,"attributes":{"topics":["trade_rules"],"geography":"RU"}}]})
            return schema.model_validate({"accepted_indexes":[0],"issues":[]})
    calls = Script()
    facts, _, _ = extract({"id":"src", "version":1,"title":"Пошлина","text":body,"received_at":"2026-10-21T00:00:00Z"},calls)
    assert calls.count == 2 and len(facts) == 1
    span = facts[0]["source_span"]
    assert body[span["start"]:span["end"]] == quote and span["start"] > 6500


def test_gap_or_unselected_quote_is_never_accepted():
    from app.intelligence.extraction import _candidates
    draft=ExtractionDraft.model_validate({"facts":[{"quote":"НЕВИДИМЫЙ ФАКТ","attributes":{"topics":["sector"]}}]})
    assert _candidates(draft,"начало НЕВИДИМЫЙ ФАКТ конец",[{"start":0,"end":6}]) == []


def test_unicode_window_budget():
    for text in ("😀"*10000, "Я"*9999, "one "*8000):
        view, coverage=source_windows(text)
        assert len(view.encode()) <= 6500
        assert coverage["windows"][-1]["end"] == len(text)


@pytest.fixture
def payment_context(tmp_path):
    out = tmp_path/"prepared"
    prepare_data(Path(__file__).resolve().parents[2]/"data/synthetic", out, top_n=10)
    rows=[json.loads(s) for s in (out/"contexts.jsonl").read_text().splitlines()]
    return out, next(r for r in rows if r["company_id"] == "S_BUILD")


def base_profile(context):
    p=context["profile"]
    return {"id":context["company_id"],"profile_version":"api-v1","industry":p["industry"],
        "region_registration":p["registration_region"],"products":p["products"],
        "fts_flag":p["fts"]["flag"],"fts_countries":p["fts"]["countries"],"features":[],"role":"cfo"}


def test_payment_context_keeps_evidence_identity_and_binds_current_profile(payment_context):
    _, context=payment_context
    enriched=attach_payment_context(base_profile(context),context)
    normalized=normalize_company(enriched)
    assert normalized["payment_sectors"]
    assert {f["id"] for f in enriched["features"]} == {f["id"] for f in context["features"] if f["name"].startswith("payment_")}
    assert all(f["version"] == context["context_version"] for f in enriched["features"])


@pytest.mark.parametrize("mutation",["company","industry","denominator","share","purpose"])
def test_payment_binding_rejects_wrong_or_unverifiable_inputs(payment_context, mutation):
    _, context=payment_context; context=deepcopy(context); base=base_profile(context)
    payment=next(f for f in context["features"] if f["name"]=="payment_outgoing")
    if mutation=="company": base["id"]="OTHER"
    elif mutation=="industry": base["industry"]={"code":"62.01","name":"Другая отрасль"}
    elif mutation=="denominator": payment["value"]["shares_denominator_raw"]="1"
    elif mutation=="share": payment["value"]["top_counterparties"][0]["share_of_all_flow"]=.01
    else: payment["value"]["limitations"]=[]
    with pytest.raises(KnowledgeError): attach_payment_context(base,context)


def test_payment_reader_checks_manifest_and_falls_back_explicitly(payment_context,monkeypatch):
    from app.intelligence.payments import enrich_payments
    out, context=payment_context; monkeypatch.setenv("COMPANY_CONTEXT_DIR",str(out))
    assert len(enrich_payments(base_profile(context))["features"])==2
    with (out/"contexts.jsonl").open("a") as f: f.write("\n ")
    result=enrich_payments(base_profile(context))
    assert result["features"]==[] and "payment_context_unavailable_or_mismatched" in result["limitations"]


def test_shared_unknown_outcome_is_not_bypassed_by_new_generation_options(store):
    from dataclasses import replace
    from test_intelligence import ScriptedGateway
    _, factory, _, settings=store
    ids=public_copies(store)
    first,_=runtime(factory,settings,ScriptedGateway(timeout=True))
    with pytest.raises(AnalysisError):
        first.extract_event(ids[0],Calls(first.provider,"unknown-options",lambda:None))
    models=dict(settings.models)
    models[settings.extraction_model]=replace(models[settings.extraction_model],output_limit=2400)
    second,gateway=runtime(factory,replace(settings,models=models))
    with pytest.raises(AnalysisError,match="extraction_outcome_unknown"):
        second.extract_event(ids[1],Calls(second.provider,"changed-options",lambda:None))
    assert gateway.seen==[]


def test_concurrent_company_waits_without_duplicate_extraction(store):
    _,factory,_,settings=store
    ids=public_copies(store)
    first,gateway=runtime(factory,settings)
    second,other_gateway=runtime(factory,settings)
    checked=[]
    def heartbeat():
        if checked: return
        checked.append(True)
        with pytest.raises(AnalysisError,match="extraction_in_progress"):
            second.extract_event(ids[1],Calls(second.provider,"concurrent",lambda:None))
    first.extract_event(ids[0],Calls(first.provider,"first-concurrent",heartbeat))
    assert checked and other_gateway.seen==[]
    second.extract_event(ids[1],Calls(second.provider,"after-first",lambda:None))
    assert other_gateway.seen==[]


def test_rss_label_without_delivery_proof_does_not_enable_sharing(store):
    client,factory,headers,settings=store
    ids=[stored_event(client,factory,headers[c]) for c in ("DEMO_A","DEMO_B")]
    with factory.begin() as db:
        for ident in ids:
            db.get(SourceDocument,db.get(Event,ident).source_id).kind="rss_article"
    engine,gateway=runtime(factory,settings)
    for i,ident in enumerate(ids):
        engine.extract_event(ident,Calls(engine.provider,f"unproven-{i}",lambda:None))
    assert sum(title=="ExtractionDraft" for title,_ in gateway.seen)==2
