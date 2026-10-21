"""Reprints, local evaluation, payment tools and immutable saved-output auditing."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest
from app.intelligence.offline import audit_database, audit_record, evaluate_corpus, main
from app.intelligence.agent import Tools
from app.intelligence.schemas import GetFeatures
from app.intelligence.storage import Insight
from app.intelligence.worker import process_next
from alphich_kb.personalization.inputs import SelectionScope
from alphich_kb.personalization.selector import NewsSelector
from test_intelligence import store, stored_event, queued_insight, runtime
from test_ai_offline_features import payment_context, base_profile
from alphich_kb.personalization.payments import attach_payment_context

FIXTURES=Path(__file__).resolve().parents[2]/"knowledge/fixtures"


def raw_event(ident, text, *, company=None, when="2026-10-20T10:00:00Z"):
    return {"id":ident,"version":1,"company_id":company,"title":"Изменения экспортной пошлины",
        "available_at":when,"published_at":when,"facts":[],
        "sources":[{"id":"src_"+ident,"version":1,"text":text,"available_at":when,"published_at":when,"url":"https://example.org/"+ident}]}


BODY="Экспортная пошлина увеличена на 12%. "+" ".join("слово"+chr(1072+i//20)+chr(1072+i%20) for i in range(100))


def visible(rows, company="A", allowed=None):
    selector=NewsSelector(rows)
    return selector._visible(SelectionScope(company,"2026-10-21T12:00:00Z",frozenset(allowed or [r["id"] for r in rows])))


def test_near_reprints_group_sources_but_keep_facts_on_original_source():
    rows=[raw_event("a",BODY),raw_event("b","Редакция сообщает: "+BODY)]
    groups, exclusions=visible(rows)
    assert len(groups)==1 and exclusions["near_reprint"]==1
    assert groups[0]["duplicate_event_ids"]==["b"]
    assert {s["source_id"] for s in groups[0]["reprint_sources"]}=={"src_a","src_b"}


@pytest.mark.parametrize("change",["number","direction","date","negation","entity"])
def test_substantive_news_update_is_not_merged(change):
    other=BODY.replace("12%","15%") if change=="number" else BODY.replace("увеличена","снижена") if change=="direction" else BODY
    if change=="negation": other=BODY.replace("увеличена","не увеличена")
    if change=="entity": other=BODY+' Компания «Другая».'
    date="2026-10-17T10:00:00Z" if change=="date" else "2026-10-20T10:00:00Z"
    groups,_=visible([raw_event("a",BODY),raw_event("b",other,when=date)])
    assert len(groups)==2


def test_reprint_sources_never_include_foreign_or_future_inputs():
    rows=[raw_event("a",BODY,company="A"),raw_event("private",BODY,company="B"),
          raw_event("future",BODY,when="2026-10-22T10:00:00Z")]
    groups, exclusions=visible(rows)
    assert len(groups)==1 and groups[0]["duplicate_event_ids"]==[]
    assert groups[0]["reprint_group_size"]==1
    assert exclusions["access_denied"]==1 and exclusions["future_source_version"]==1


def test_payment_tool_and_sector_selection_use_original_feature_ids(payment_context):
    _, context=payment_context
    enriched=attach_payment_context(base_profile(context),context)
    sector=next(s for f in context["features"] if f["name"]=="payment_outgoing" for s in f["value"]["known_sectors"])
    event=raw_event("sector","Изменение условий работы отрасли контрагента.",company=context["company_id"])
    text=event["sources"][0]["text"]
    event["facts"]=[{"id":"fact","source_id":"src_sector","source_version":1,"text":text,
        "source_span":{"start":0,"end":len(text)},"source_available_at":event["available_at"],
        "verification_status":"human_verified","attributes":{"topics":["sector"],"geography":"RU","industry_codes":[sector["code"]]}}]
    scope=SelectionScope(context["company_id"],"2026-10-21T12:00:00Z",frozenset({"sector"}))
    selector=NewsSelector([event]); result=selector.select(enriched,scope,mode="lexical")
    item=result["items"][0]
    assert "payment_neighbor_sector" in item["reason_codes"]
    tools=Tools(enriched,selector.events[0],item,scope,None,[])
    action=GetFeatures(action="get_payment_neighbors")
    response=tools.invoke(action)
    assert len(response["evidence"])==2 and response["limitations"]==[]
    assert all(e["company_id"]==context["company_id"] for e in response["evidence"])
    assert all("person" not in e["name"] for e in response["evidence"])


def test_evaluation_rejects_duplicate_and_incomplete_label_matrix(tmp_path):
    events={"events":[raw_event("one",BODY)]}
    profiles=json.loads((FIXTURES/"company_profiles.json").read_text())
    paths=[tmp_path/n for n in ("profiles.json","events.json","labels.json")]
    paths[0].write_text(json.dumps(profiles)); paths[1].write_text(json.dumps(events))
    labels={"company_ids":["S_BUILD"],"as_of":"2026-10-21T12:00:00Z","labels":[]}
    paths[2].write_text(json.dumps(labels))
    with pytest.raises(ValueError,match="complete"): evaluate_corpus(*paths)
    labels["labels"]=[{"company_id":"S_BUILD","event_id":"one","grade":2}]*2
    paths[2].write_text(json.dumps(labels))
    with pytest.raises(ValueError,match="duplicate"): evaluate_corpus(*paths)


def test_evaluation_is_explicit_about_unreviewed_labels_and_no_generation(tmp_path,monkeypatch):
    import socket
    def blocked(*args,**kwargs): raise AssertionError("Offline evaluation attempted network")
    monkeypatch.setattr(socket,"create_connection",blocked)
    profiles=FIXTURES/"company_profiles.json"
    events=tmp_path/"events.json"; events.write_text(json.dumps({"events":[raw_event("one","Банк России сохранил ключевую ставку.")]}))
    labels=tmp_path/"labels.json"; labels.write_text(json.dumps({"company_ids":["S_BUILD"],"as_of":"2026-10-21T12:00:00Z",
        "labels":[{"company_id":"S_BUILD","event_id":"one","grade":2}]}))
    r=evaluate_corpus(profiles,events,labels)
    assert r["generative_calls"]==0 and r["independently_reviewed"] is False
    assert len(r["modes"]["lexical"]["results"])==2


def test_audit_uses_read_only_database_and_detects_tampered_quotes(store,tmp_path,monkeypatch):
    client,factory,headers,settings=store
    event=stored_event(client,factory,headers["DEMO_A"])
    job=queued_insight(client,headers["DEMO_A"],event).json()["id"]
    engine,_=runtime(factory,settings)
    assert process_next(factory,engine)
    ident=client.get("/api/v1/jobs/"+job,headers=headers["DEMO_A"]).json()["result_id"]
    path=Path(factory.kw["bind"].url.database)
    before=hashlib.sha256(path.read_bytes()).hexdigest()
    import socket
    monkeypatch.setattr(socket,"create_connection",lambda *a,**k:pytest.fail("Audit opened network"))
    report=audit_database(path)
    assert report["checked"]==1 and report["failed"]==0
    assert hashlib.sha256(path.read_bytes()).hexdigest()==before
    with factory() as db:
        record=db.get(Insight,ident)
        evidence=deepcopy(record.evidence)
        news=next(e for e in evidence.values() if e["kind"]=="event_fact")
        news["text"]="Подменённая цитата, отсутствующая в источнике."
        assert "news_quote_span_mismatch" in audit_record(record.payload,evidence,record.snapshot)
        frozen=deepcopy(record.snapshot)
        frozen["profile"]["role"]="ceo"
        assert "snapshot_role_mismatch" in audit_record(record.payload,record.evidence,frozen)


def test_raw_it_cabling_is_not_civil_engineering_evidence():
    profiles=json.loads((FIXTURES/"company_profiles.json").read_text())["companies"]
    profile=deepcopy(next(p for p in profiles if p["company_id"]=="S_BUILD"))
    # Prepared synthetic construction profile uses OKVED 41; use a bound 42 profile.
    profile["profile"]["industry"]={"code":"42.99","name":"Строительство инженерных сооружений"}
    next(f for f in profile["features"] if f["name"]=="industry")["value"]=profile["profile"]["industry"]
    event=raw_event("network","Модернизирована сетевая инфраструктура и кабельная система промышленного предприятия.")
    event["title"]="Модернизация компьютерной сети"
    result=NewsSelector([event]).select(profile,SelectionScope("S_BUILD","2026-10-21T12:00:00Z",frozenset({"network"})),mode="lexical")
    assert result["items"]==[]


def test_developer_news_is_only_an_extraction_candidate_without_machine_facts():
    profiles=json.loads((FIXTURES/"company_profiles.json").read_text())["companies"]
    profile=next(p for p in profiles if p.get("company_id",p.get("id"))=="S_SOFTWARE")
    event=raw_event("dev","Программисты обсуждают сопровождение свободного ПО и инструменты разработки.")
    result=NewsSelector([event]).select(profile,SelectionScope("S_SOFTWARE","2026-10-21T12:00:00Z",frozenset({"dev"})),mode="lexical")
    assert result["items"][0]["status"]=="needs_extraction"
    assert result["items"][0]["matched_fact_ids"]==[]


def test_collection_uses_configured_fetcher_without_db_or_llm(monkeypatch,tmp_path):
    from app.rss_fetch import Download
    from app.rss_parse import FeedItem
    from app.intelligence.offline import collect_corpus
    from app.rss import FeedConfig
    import app.rss_fetch, app.rss_parse, app.rss
    feed=FeedConfig(id="fixture",url="https://example.org/rss",article_hosts=["example.org"])
    monkeypatch.setattr(app.rss,"load_config",lambda _: [feed])
    calls=[]
    def get(self,url,hosts,validators=None):
        calls.append((url,hosts))
        return Download(url,200,{"content-type":"text/html"},b"fixture")
    monkeypatch.setattr(app.rss_fetch.PublicFetcher,"get",get)
    monkeypatch.setattr(app.rss_parse,"parse_feed",lambda *_:[FeedItem("Заголовок новости","https://example.org/article","2026-10-20T10:00:00Z")])
    monkeypatch.setattr(app.rss_parse,"article_text",lambda *_:"Полный извлечённый текст публикации. "*5)
    monkeypatch.chdir(tmp_path)
    result=collect_corpus("unused","data/processed/corpus.json",1,1)
    assert result["collected"]==1 and result["generative_calls"]==0 and len(calls)==2
    saved=json.loads(Path("data/processed/corpus.json").read_text())
    assert saved["events"][0]["facts"]==[] and saved["events"][0]["company_id"] is None


def test_frozen_corpus_hash_must_match_labels(tmp_path):
    profiles=FIXTURES/"company_profiles.json"
    events=tmp_path/"events.json"; events.write_text(json.dumps({"events":[raw_event("one",BODY)]}))
    labels=tmp_path/"labels.json"; labels.write_text(json.dumps({"events_sha256":"bad"}))
    with pytest.raises(ValueError,match="frozen_label_corpus_changed"):
        evaluate_corpus(profiles,events,labels)


def test_prepared_payments_enter_queue_snapshot_and_survive_file_changes(store,payment_context,monkeypatch):
    from app.core import Company,profile_payload
    from app.intelligence.storage import AnalysisTask
    client,factory,_,settings=store
    out,context=payment_context
    p=context["profile"]; cid=context["company_id"]
    with factory.begin() as db:
        db.add(Company(**profile_payload(cid,p["source_fields"],[p["source_record"]+1])))
    monkeypatch.setenv("DEMO_COMPANY_IDS","DEMO_A,DEMO_B,"+cid)
    monkeypatch.setenv("COMPANY_CONTEXT_DIR",str(out))
    session=client.post("/api/v1/demo-session",json={"company_id":cid,"role":"cfo"}).json()
    headers={"X-Demo-Session":session["token"]}
    event=stored_event(client,factory,headers)
    response=client.post(f"/api/v1/companies/{cid}/insights",headers={**headers,"Idempotency-Key":"payment-queue-001"},
        json={"event_id":event})
    assert response.status_code==202,response.text
    job=response.json()["id"]
    with factory() as db:
        profile=db.get(AnalysisTask,job).profile
        assert len([f for f in profile["features"] if f["name"].startswith("payment_")])==2
    (out/"contexts.jsonl").write_text("invalid after enqueue")
    engine,_=runtime(factory,settings)
    assert process_next(factory,engine)
    result=client.get("/api/v1/jobs/"+job,headers=headers).json()
    assert result["status"]=="succeeded",result
    with factory() as db:
        saved=db.get(Insight,result["result_id"])
        assert saved.snapshot["profile"]["payment_context_version"]==context["context_version"]
