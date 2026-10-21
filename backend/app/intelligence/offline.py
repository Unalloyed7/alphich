"""Unpaid collection, selection evaluation and read-only saved-output audits.

No Settings, dotenv, Provider.generate or gateway endpoints are used here.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import statistics
from urllib.parse import quote


def save_local(path, data):
    path = Path(path)
    if not path.resolve().is_relative_to(Path("data/processed").resolve()):
        raise ValueError("output_must_be_ignored_processed_data")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".partial")
    with temporary.open("w", encoding="utf-8") as stream:
        temporary.chmod(0o600)
        json.dump(data, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def collect_corpus(config, output, limit=20, per_feed=5):
    from app.rss import load_config
    from app.rss_fetch import PublicFetcher, CollectionError, public_url
    from app.rss_parse import discover_items, article_text
    if not 1 <= limit <= 100 or not 1 <= per_feed <= 20:
        raise ValueError("invalid_corpus_limits")
    if not Path(output).resolve().is_relative_to(Path("data/processed").resolve()):
        raise ValueError("output_must_be_ignored_processed_data")
    events, errors, seen = [], [], set()
    collected = datetime.now(timezone.utc).isoformat()
    for feed in load_config(config):
        if len(events) >= limit: break
        try:
            fetcher = PublicFetcher()
            response = fetcher.get(feed.url, feed.hosts)
            if response.status != 200: raise CollectionError("feed_http_"+str(response.status))
            candidates = discover_items(response.body, response.url, feed.format)
            accepted = attempts = 0
            for item in candidates:
                if len(events) >= limit or accepted >= per_feed: break
                try:
                    public_url(item.url, feed.hosts)
                    if not feed.accepts_path(item.url) or item.url in seen: continue
                    if attempts >= max(15,2*per_feed): break
                    attempts+=1
                    article = fetcher.get(item.url, feed.hosts)
                    if article.status != 200: raise CollectionError("article_http_"+str(article.status))
                    if not feed.accepts_path(article.url): continue
                    text = article_text(article.body, article.url, article.headers.get("content-type", ""))
                    text_hash = hashlib.sha256(text.encode()).hexdigest()
                    if text_hash in seen: continue
                    ident = "rss_" + hashlib.sha256(item.url.encode()).hexdigest()[:20]
                    events.append({"id":ident,"version":1,"company_id":None,"title":item.title,
                        "published_at":item.published_at,"available_at":collected,"facts":[],
                        "sources":[{"id":"source_"+ident,"version":1,"title":item.title,"text":text,
                            "url":article.url,"available_at":collected,"published_at":item.published_at,
                            "completeness":"extracted_article_text"}],
                        "provenance":{"feed_id":feed.id,"feed_url":feed.url,"fetched_at":collected,
                            "requested_url":item.url,"text_sha256":text_hash}})
                    seen.update((item.url,text_hash)); accepted+=1
                except CollectionError as error:
                    errors.append({"feed_id":feed.id,"code":error.code})
        except CollectionError as error:
            errors.append({"feed_id":feed.id,"code":error.code})
    payload={"version":"real-rss-corpus-v1","scope":"public_rss_articles_no_generated_facts",
        "collected_at":collected,"generative_calls":0,"events":events,"errors":errors}
    save_local(output,payload)
    return {"collected":len(events),"requested":limit,"errors":len(errors),"output":str(output),"generative_calls":0}


def evaluate_corpus(profiles_path, events_path, labels_path, encoder=None):
    from alphich_kb.common import file_hash, read_json
    from alphich_kb.personalization.cli import load_rows
    from alphich_kb.personalization.inputs import SelectionScope
    from alphich_kb.personalization.selector import NewsSelector
    from alphich_kb.personalization.evaluate import ndcg
    profiles={p.get("company_id",p.get("id")):p for p in load_rows(profiles_path,"companies")}
    events=load_rows(events_path,"events"); labels=read_json(labels_path)
    scale = labels.get("grade_scale", {"min":0,"max":3})
    threshold = labels.get("selection_threshold", 2)
    maximum = scale.get("max")
    if scale.get("min") != 0 or type(maximum) is not int or maximum not in {3,5} or type(threshold) is not int or not 1 <= threshold <= maximum:
        raise ValueError("invalid_grade_scale_or_threshold")
    if maximum == 5 and threshold != 4:
        raise ValueError("0_to_5_rubric_selects_only_4_and_5")
    if labels.get("events_sha256") and labels["events_sha256"]!=file_hash(events_path):
        raise ValueError("frozen_label_corpus_changed")
    companies=labels["company_ids"]; roles=labels.get("roles",["cfo","ceo"])
    if len(companies)!=len(set(companies)) or set(companies)-profiles.keys() or not companies or set(roles)-{"cfo","ceo"}:
        raise ValueError("invalid_label_scope")
    grades={}
    event_ids={e["id"] for e in events}
    for row in labels["labels"]:
        key=(row["company_id"],row["event_id"])
        if key in grades or type(row["grade"]) is not int or not 0<=row["grade"]<=maximum:
            raise ValueError("invalid_or_duplicate_grade")
        if maximum == 5:
            from alphich_kb.personalization.grading import validate_assessment
            selected_by_rubric = validate_assessment(row["grade"],row["assessment"])
            if "selected" in row and row["selected"] is not selected_by_rubric:
                raise ValueError("selected_flag_disagrees_with_grade")
        grades[key]=row["grade"]
    if set(grades)!={(c,e) for c in companies for e in event_ids}:
        raise ValueError("label_matrix_must_be_complete")
    selector=NewsSelector(events,encoder); reports={}
    for mode in (["lexical","dense","hybrid"] if encoder else ["lexical"]):
        rows=[]
        for company in companies:
            for role in roles:
                scope=SelectionScope(company,labels["as_of"],frozenset(event_ids),role)
                result=selector.select(profiles[company],scope,mode=mode)
                if result["mode"]!=mode: raise ValueError("evaluation_cannot_hide_dense_fallback")
                selected=result["items"]
                selected_ids={i["event_id"] for i in selected}
                grouped_ids={ident for i in selected for ident in i["duplicate_event_ids"]}
                relevant={e for e in event_ids if grades[company,e]>=threshold}
                selected_grades=[grades[company,i["event_id"]] for i in selected]
                rows.append({"company_id":company,"role":role,
                    "ndcg_at_5":ndcg(selected_grades,[grades[company,e] for e in event_ids]),
                    "precision_at_5":sum(g>=threshold for g in selected_grades)/5,
                    "recall_at_5":len(relevant & (selected_ids|grouped_ids))/len(relevant) if relevant else None,
                    "irrelevant_selected":sum(g==0 for g in selected_grades),
                    "below_selection_threshold":sum(g<threshold for g in selected_grades),
                    "missed_relevant_event_ids":sorted(relevant-selected_ids-grouped_ids),
                    "items":[{"event_id":i["event_id"],"grade":grades[company,i["event_id"]],
                        "status":i["status"],"reason_codes":i["reason_codes"]} for i in selected],
                    "review_candidate_ids":[i["event_id"] for i in result["review_queue"]],
                    "elapsed_ms":result["trace"]["elapsed_ms"]})
        scores=[r["ndcg_at_5"] for r in rows if r["ndcg_at_5"] is not None]
        reports[mode]={"mean_ndcg_at_5":statistics.mean(scores) if scores else None,
            "mean_precision_at_5":statistics.mean(r["precision_at_5"] for r in rows),
            "irrelevant_selected":sum(r["irrelevant_selected"] for r in rows),
            "below_selection_threshold":sum(r["below_selection_threshold"] for r in rows),
            "p50_ms":statistics.median(r["elapsed_ms"] for r in rows),"results":rows}
    return {"version":"real-news-selection-report-v2","scope":labels.get("scope","unspecified"),
        "grade_scale":scale,"selection_threshold":threshold,
        "label_review_status":labels.get("review_status","unreviewed"),
        "independently_reviewed":labels.get("independently_reviewed") is True,
        "label_pairs":len(grades),"roles":roles,"as_of":labels["as_of"],
        "profiles_sha256":file_hash(profiles_path),"events_sha256":file_hash(events_path),"labels_sha256":file_hash(labels_path),
        "embedding_space_id":encoder.space_id if encoder else None,"generative_calls":0,
        "limitations":["author_labels_require_independent_human_review","raw_news_candidates_are_not_verified_facts",
            "selection_metrics_do_not_measure_generated_answer_quality"],"modes":reports}


def audit_record(card, evidence, snapshot):
    from .evidence import verify_draft, company_evidence
    from .schemas import CardDraft
    from alphich_kb.personalization.inputs import instant
    issues=[]
    try:
        if card.get("claims"):
            draft=CardDraft.model_validate({k:card[k] for k in CardDraft.model_fields if k in card})
            issues.extend(verify_draft(draft,evidence))
        elif card.get("status")!="insufficient_evidence":
            issues.append("missing_claims")
        as_of=instant(card["as_of"])
        profile=snapshot["profile"]; event=snapshot["event"]
        if profile.get("id",profile.get("company_id")) != card["company_id"]:
            issues.append("snapshot_company_mismatch")
        if profile.get("role") not in {None,card.get("role")}:
            issues.append("snapshot_role_mismatch")
        if card.get("profile_version") and card["profile_version"]!=profile.get("profile_version"):
            issues.append("snapshot_profile_version_mismatch")
        expected_company=company_evidence(profile)
        sources={s["id"]:s for s in event["sources"]}
        for ident,e in evidence.items():
            kind=e.get("kind")
            if kind in {"event_fact","company_feature"} and e.get("company_id")!=card["company_id"]:
                issues.append("evidence_company_mismatch")
            if kind=="company_feature" and (ident not in expected_company or expected_company[ident]["text"]!=e.get("text")):
                issues.append("company_feature_snapshot_mismatch")
            for field in ("published_at","source_available_at"):
                value=e.get(field)
                if value:
                    if len(value)==10:
                        future=datetime.strptime(value,"%Y-%m-%d").date()>as_of.date()
                    else:
                        future=instant(value)>as_of
                    if future: issues.append("future_evidence")
            if kind=="event_fact":
                source=sources.get(e.get("source_id"))
                # Related event evidence can be sourced from the frozen related set.
                if source is None:
                    source=next((s for related in snapshot.get("related_events",[]) for s in related["sources"] if s["id"]==e.get("source_id")),None)
                span=e.get("source_span",{})
                start,end=span.get("start"),span.get("end")
                if not source or type(start) is not int or type(end) is not int or not 0<=start<end<=len(source["text"]) or source["text"][start:end]!=e.get("text"):
                    issues.append("news_quote_span_mismatch")
        if card.get("bank_view",{}).get("status")=="not_found" and any(c["kind"]=="bank_view" for c in card["claims"]):
            issues.append("bank_status_contradicts_claim")
    except (KeyError,TypeError,ValueError,AttributeError):
        issues.append("invalid_saved_record")
    return sorted(set(issues))


def audit_database(path):
    path=Path(path).resolve()
    db=sqlite3.connect("file:"+quote(str(path),safe="/")+"?mode=ro",uri=True)
    db.execute("PRAGMA query_only=ON")
    rows=[]
    try:
        insights={}
        for ident,company,role,payload,evidence,snapshot in db.execute("SELECT id,company_id,role,payload,evidence,snapshot FROM insights"):
            card,registry,frozen=json.loads(payload),json.loads(evidence),json.loads(snapshot)
            insights[ident]=(card,frozen)
            issues=audit_record(card,registry,frozen)
            if card.get("company_id")!=company or card.get("role")!=role:
                issues.append("record_owner_scope_mismatch")
            rows.append({"kind":"insight","id":ident,"issues":issues})
        for ident,parent_id,company,role,payload,evidence in db.execute("SELECT id,insight_id,company_id,role,payload,evidence FROM insight_answers"):
            answer=json.loads(payload); parent=insights.get(parent_id)
            if parent is None:
                issues=["parent_insight_missing"]
            else:
                card,frozen=parent
                scoped={**answer,"as_of":card["as_of"],"company_id":card["company_id"],"role":card["role"],
                    "status":answer.get("status","ready_with_conditions")}
                issues=audit_record(scoped,json.loads(evidence),frozen)
                if answer.get("company_id",card["company_id"])!=card["company_id"] or answer.get("role",card["role"])!=card["role"]:
                    issues.append("answer_owner_mismatch")
                if company!=card["company_id"] or role!=card["role"]:
                    issues.append("record_owner_scope_mismatch")
            rows.append({"kind":"answer","id":ident,"issues":sorted(set(issues))})
        ledger=db.execute("SELECT COUNT(*),COALESCE(SUM(cost_micros),0),COALESCE(SUM(CASE WHEN cost_micros IS NULL THEN reserved_micros ELSE 0 END),0) FROM llm_calls").fetchone()
    finally:
        db.close()
    return {"version":"saved-ai-audit-v1","database_opened_read_only":True,"generative_calls":0,
        "checked":len(rows),"passed":sum(not r["issues"] for r in rows),"failed":sum(bool(r["issues"]) for r in rows),
        "ledger":{"calls":ledger[0],"known_cost_micros":ledger[1],"unresolved_reserve_micros":ledger[2],
            "unit":"existing_ledger_accounting_units_no_currency_reinterpretation"},
        "limitations":["structural_audit_does_not_prove_semantic_correctness"],"results":rows}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest="command",required=True)
    c=sub.add_parser("collect"); c.add_argument("--config",default="backend/app/rss_sources.json")
    c.add_argument("--limit",type=int,default=20); c.add_argument("--per-feed",type=int,default=5)
    c.add_argument("--output",required=True)
    e=sub.add_parser("evaluate"); e.add_argument("--profiles",required=True); e.add_argument("--events",required=True)
    e.add_argument("--labels",required=True); e.add_argument("--output",required=True)
    e.add_argument("--dense",action="store_true"); e.add_argument("--model-dir",default="data/models/e5-base")
    e.add_argument("--model-manifest",default="knowledge/embedding-model.json")
    a=sub.add_parser("audit"); a.add_argument("--database"); a.add_argument("--report",action="append",default=[])
    a.add_argument("--output",required=True)
    args=p.parse_args(argv)
    try:
        if args.command=="collect":
            result=collect_corpus(args.config,args.output,args.limit,args.per_feed)
            print(json.dumps(result,ensure_ascii=False)); return 0 if result["collected"]==args.limit else 2
        if args.command=="evaluate":
            encoder=None
            if args.dense:
                from alphich_kb.embeddings import E5Encoder
                encoder=E5Encoder(args.model_dir,args.model_manifest,threads=2)
            result=evaluate_corpus(args.profiles,args.events,args.labels,encoder)
        else:
            if not args.database and not args.report: p.error("audit requires --database or --report")
            result={"generative_calls":0}
            if args.database: result["database"]=audit_database(args.database)
            if args.report:
                from .evaluation import recheck_report
                result["reports"]=[{"path":path,"result":recheck_report(path)} for path in args.report]
        save_local(args.output,result)
        print(json.dumps({"status":"completed","output":args.output,"generative_calls":0,
            "failed_records":result.get("database",{}).get("failed",0)},ensure_ascii=False))
        failed_reports=any(not r["result"]["all_selected_controls_passed"] for r in result.get("reports",[]))
        return 2 if result.get("database",{}).get("failed",0) or failed_reports else 0
    except (OSError,ValueError,KeyError,TypeError,sqlite3.Error):
        print(json.dumps({"status":"failed","error":"invalid_offline_input","generative_calls":0})); return 2


if __name__=="__main__":
    raise SystemExit(main())
