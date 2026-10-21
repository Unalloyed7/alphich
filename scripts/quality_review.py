#!/usr/bin/env python3
"""Offline, local-only, two-reader blind news-grade review. Python stdlib only.

No LLM, no remote fetches, no modification to Alphich production databases.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import re
import secrets
import sqlite3
import sys
import threading
from datetime import datetime, timezone
from urllib.parse import urlsplit, parse_qs

HERE = Path(__file__).resolve().parent
FIXTURE_NAMES = ("real_news_selection_labels.v2.json", "real_news_selection_labels.v3.json")
SCHEMA_VERSION = 1
GEOGRAPHY = {"national", "global_market", "confirmed_operations", "registration_only", "other_region", "unknown", "not_applicable"}
SCALE = {"material", "limited", "unknown", "none"}
APPLICABILITY = {"confirmed", "conditional", "not_confirmed", "unknown"}
DB_LOCK = threading.RLock()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(str(path), timeout=15)
    db.row_factory = sqlite3.Row
    db.execute("PRAGMA foreign_keys=ON")
    db.execute("PRAGMA busy_timeout=15000")
    return db


@contextmanager
def managed(path: Path):
    db = connect(path)
    try:
        with db:
            yield db
    finally:
        db.close()


def canon(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def prepare_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("DB уже существует; повторный init запрещён. Используйте serve или отдельный путь.")
    db = connect(path)
    try:
        db.executescript("""
          CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
          CREATE TABLE users (
            username TEXT PRIMARY KEY, role TEXT NOT NULL CHECK(role IN ('reviewer','arbiter')),
            token_hash TEXT NOT NULL UNIQUE, created_at TEXT NOT NULL
          );
          CREATE TABLE pairs (
            id INTEGER PRIMARY KEY, dataset TEXT NOT NULL, dataset_version TEXT NOT NULL,
            fixture_hash TEXT NOT NULL, event_id TEXT NOT NULL, company_id TEXT NOT NULL,
            title TEXT NOT NULL, url TEXT, published_at TEXT, article_text TEXT,
            company_context TEXT, original_prediction_json TEXT NOT NULL,
            UNIQUE(dataset,event_id,company_id)
          );
          CREATE TABLE reviews (
            id INTEGER PRIMARY KEY, pair_id INTEGER NOT NULL REFERENCES pairs(id),
            reviewer TEXT NOT NULL REFERENCES users(username), revision INTEGER NOT NULL,
            grade INTEGER NOT NULL CHECK(grade BETWEEN 0 AND 5), justification TEXT NOT NULL,
            geography TEXT NOT NULL, scale TEXT NOT NULL, applicability TEXT NOT NULL,
            geographic_limit TEXT NOT NULL, scale_limit TEXT NOT NULL, applicability_limit TEXT NOT NULL,
            created_at TEXT NOT NULL, UNIQUE(pair_id,reviewer,revision)
          );
          CREATE TABLE adjudications (
            id INTEGER PRIMARY KEY, pair_id INTEGER NOT NULL REFERENCES pairs(id),
            arbiter TEXT NOT NULL REFERENCES users(username), revision INTEGER NOT NULL,
            grade INTEGER NOT NULL CHECK(grade BETWEEN 0 AND 5), justification TEXT NOT NULL,
            geography TEXT NOT NULL, scale TEXT NOT NULL, applicability TEXT NOT NULL,
            geographic_limit TEXT NOT NULL, scale_limit TEXT NOT NULL, applicability_limit TEXT NOT NULL,
            review_ids_json TEXT NOT NULL, created_at TEXT NOT NULL,
            UNIQUE(pair_id, revision)
          );
          CREATE INDEX ix_review_pair ON reviews(pair_id,reviewer,revision);
          CREATE INDEX ix_adjudication_pair ON adjudications(pair_id,revision);
        """)
        db.execute("INSERT INTO metadata VALUES (?,?)", ("schema_version", str(SCHEMA_VERSION)))
        db.execute("INSERT INTO metadata VALUES (?,?)", ("initialized_at", now()))
        db.commit()
    except BaseException:
        db.close()
        path.unlink(missing_ok=True)
        raise
    db.close()
    path.chmod(0o600)


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Ожидался JSON-объект: {path}")
    return value


def enrich_data(db: sqlite3.Connection, articles: Path | None, contexts: Path | None) -> tuple[int,int]:
    """Sidecars are independent of the AI fixture labels, and private to the operator.

    Articles: event_id => {text, published_at}; contexts: company_id =>
    {features:[{name,value,source_ref}], verified_at, note}.
    """
    article_map, context_map = load_json(articles) if articles else {}, load_json(contexts) if contexts else {}
    if not article_map and not context_map:
        return 0, 0
    if db.execute("SELECT count(*) FROM reviews").fetchone()[0] or db.execute("SELECT count(*) FROM adjudications").fetchone()[0]:
        raise ValueError("После начала разметки менять исходные материалы нельзя. Создайте новую БД.")
    count_articles, count_contexts = 0, 0
    for event_id, entry in article_map.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("text"), str) or not entry["text"].strip():
            raise ValueError("В articles требуются объект и непустой text для " + str(event_id))
        if len(entry["text"]) > 1000000:
            raise ValueError("Слишком длинная статья")
        when = entry.get("published_at")
        if when is not None and (not isinstance(when, str) or len(when) > 80):
            raise ValueError("Некорректный published_at")
        cursor = db.execute("UPDATE pairs SET article_text=?, published_at=COALESCE(?,published_at) WHERE event_id=?",
                            (entry["text"], when, event_id))
        count_articles += cursor.rowcount
    for company_id, entry in context_map.items():
        if (not isinstance(entry, dict) or not isinstance(entry.get("features"), list)
            or not entry.get("verified_at") or not all(isinstance(f, dict) and
                   isinstance(f.get("name"), str) and "value" in f and f.get("source_ref") for f in entry["features"])):
            raise ValueError("Для подтверждённого контекста нужны features с source_ref и verified_at: " + str(company_id))
        raw = canon(entry)
        if len(raw) > 250000:
            raise ValueError("Слишком большой контекст компании")
        cursor = db.execute("UPDATE pairs SET company_context=? WHERE company_id=?", (raw, company_id))
        count_contexts += cursor.rowcount
    return count_articles, count_contexts


def initialize(db_path: Path, fixtures_dir: Path, reviewers: list[str], arbiter: str,
               articles: Path | None = None, contexts: Path | None = None) -> dict:
    if len(reviewers) != 2 or len(set(reviewers)) != 2:
        raise ValueError("Нужно ровно два различных проверяющих")
    names = [*reviewers, arbiter]
    if len(set(names)) != 3 or any(not re.fullmatch(r"[a-zA-Z][a-zA-Z0-9_.-]{1,39}", name) for name in names):
        raise ValueError("Имена должны быть уникальными (латиница, 2–40 символов)")
    fixture_data = []
    seen = set()
    for fixture in FIXTURE_NAMES:
        filename = fixtures_dir / fixture
        if not filename.is_file():
            raise ValueError("Не найден файл " + str(filename))
        raw = filename.read_bytes()
        data = json.loads(raw)
        if not isinstance(data.get("sources"), list) or not isinstance(data.get("labels"), list):
            raise ValueError("Некорректная структура " + fixture)
        dataset = fixture.replace("real_news_selection_labels.", "").replace(".json", "")
        sources = {s["event_id"]: s for s in data["sources"]}
        if len(sources) != len(data["sources"]):
            raise ValueError("Дубли event_id в " + fixture)
        rows = []
        for label in data["labels"]:
            company, event = label["company_id"], label["event_id"]
            key = (dataset, event, company)
            if key in seen or event not in sources or not isinstance(label.get("grade"), int) or label["grade"] not in range(6):
                raise ValueError("Некорректная или повторная пара " + repr(key))
            seen.add(key)
            source = sources[event]
            url = source.get("url")
            if url is not None and (not isinstance(url,str) or urlsplit(url).scheme != "https"):
                raise ValueError("У источника должен быть HTTPS-адрес либо null")
            prediction = {k: label.get(k) for k in ("grade", "rationale", "assessment", "selected")}
            rows.append((dataset, str(data.get("version", dataset)), sha256(raw), event, company,
                         str(source.get("title") or source.get("short_title") or event), url,
                         source.get("published_at"), None, None, canon(prediction)))
        fixture_data.append((dataset, rows, sha256(raw)))
    prepare_db(db_path)
    generated = {}
    try:
        with managed(db_path) as db:
            for name, role in [(reviewers[0],"reviewer"),(reviewers[1],"reviewer"),(arbiter,"arbiter")]:
                token = secrets.token_urlsafe(32)
                generated[name] = token
                db.execute("INSERT INTO users VALUES (?,?,?,?)", (name,role,sha256(token.encode()),now()))
            for dataset, rows, hash_ in fixture_data:
                db.executemany("INSERT INTO pairs(dataset,dataset_version,fixture_hash,event_id,company_id,title,url,published_at,article_text,company_context,original_prediction_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)", rows)
                db.execute("INSERT INTO metadata VALUES (?,?)", (dataset+"_sha256", hash_))
            db.execute("INSERT INTO metadata VALUES (?,?)",("fixture_pair_count",str(len(seen))))
            enrich_data(db,articles,contexts)
    except BaseException:
        db_path.unlink(missing_ok=True)
        raise
    return {"users": generated, "pairs": len(seen), "db": str(db_path)}


def latest_review_rows(db, pair_id: int) -> list[sqlite3.Row]:
    return db.execute("""SELECT r.* FROM reviews r JOIN
          (SELECT reviewer,MAX(revision) AS revision FROM reviews WHERE pair_id=? GROUP BY reviewer) k
          ON r.reviewer=k.reviewer AND r.revision=k.revision AND r.pair_id=?
          ORDER BY r.reviewer""", (pair_id, pair_id)).fetchall()


def latest_adjudication(db,pair_id: int) -> sqlite3.Row | None:
    return db.execute("SELECT * FROM adjudications WHERE pair_id=? ORDER BY revision DESC LIMIT 1",(pair_id,)).fetchone()


def agreement(a: sqlite3.Row,b:sqlite3.Row) -> bool:
    return all(a[f] == b[f] for f in ("grade", "geography", "scale", "applicability"))


def validate_review(data: dict) -> dict:
    if not isinstance(data, dict) or type(data.get("grade")) is not int or data["grade"] not in range(6):
        raise ValueError("Оценка должна быть целым числом 0–5")
    result = {"grade": data["grade"]}
    for field, options in (("geography",GEOGRAPHY),("scale",SCALE),("applicability",APPLICABILITY)):
        if data.get(field) not in options:
            raise ValueError("Некорректное значение " + field)
        result[field] = data[field]
    for field in ("justification","geographic_limit","scale_limit","applicability_limit"):
        value = data.get(field)
        if not isinstance(value, str) or not 4 <= len(value.strip()) <= 2000:
            raise ValueError("Заполните " + field + " (4–2000 символов)")
        result[field] = value.strip()
    return result


def reviewed_view(row: sqlite3.Row) -> dict:
    return {k:row[k] for k in ("id","reviewer","arbiter","grade","justification","geography","scale","applicability",
                                "geographic_limit","scale_limit","applicability_limit","revision","created_at") if k in row.keys()}


def authenticated(db, token: str) -> sqlite3.Row | None:
    if not isinstance(token,str) or not 20 <= len(token) <= 200:
        return None
    digest=sha256(token.encode())
    for row in db.execute("SELECT username,role,token_hash FROM users"):
        if hmac.compare_digest(digest,row["token_hash"]):
            return row
    return None


def state_of(db, pair_id: int, role: str, username: str) -> dict:
    rows = latest_review_rows(db,pair_id)
    adj = latest_adjudication(db,pair_id)
    if role == "reviewer":
        own = next((r for r in rows if r["reviewer"] == username), None)
        return {"state":"saved" if own else "todo", "own":reviewed_view(own) if own else None}
    if len(rows) < 2:
        state = "awaiting_reviews"
    elif adj:
        state = "adjudicated"
    else:
        state = "agreed" if agreement(*rows) else "disagreement"
    return {"state":state,"review_count":len(rows), "adjudicated":bool(adj)}


def summary(db, user: sqlite3.Row):
    pairs = db.execute("SELECT id,dataset,company_id,event_id,title FROM pairs ORDER BY dataset,event_id,company_id").fetchall()
    rows = []
    for row in pairs:
        status = state_of(db,row["id"],user["role"],user["username"])
        rows.append({"id":row["id"],"dataset":row["dataset"],"event_id":row["event_id"],
                    "company_id":row["company_id"],"title":row["title"],"state":status["state"]})
    counts = {}
    for row in rows:
        counts[row["state"]] = counts.get(row["state"],0)+1
    return {"user":user["username"],"role":user["role"],"total":len(rows),"counts":counts,"items":rows}


def case_view(db, user: sqlite3.Row, pair_id: int):
    row=db.execute("SELECT * FROM pairs WHERE id=?",(pair_id,)).fetchone()
    if row is None:
        raise KeyError("Пара не найдена")
    result={k:row[k] for k in ("id","dataset","dataset_version","event_id","company_id","title","url","published_at","article_text")}
    result["company_context"] = json.loads(row["company_context"]) if row["company_context"] else None
    result["published_at_missing"] = row["published_at"] is None
    result["article_text_missing"] = row["article_text"] is None
    result["context_missing"] = row["company_context"] is None
    result.update(state_of(db,pair_id,user["role"],user["username"]))
    if user["role"] == "arbiter" and result["review_count"] == 2:
        result["reviews"]=[reviewed_view(r) for r in latest_review_rows(db,pair_id)]
        final=latest_adjudication(db,pair_id)
        result["adjudication"] = reviewed_view(final) if final else None
    # This route NEVER exposes original_prediction_json, source grades or rationale.
    return result


def save_review(db, user:sqlite3.Row, pair_id: int, data: dict) -> dict:
    if user["role"] != "reviewer":
        raise PermissionError("Только эксперт может отправлять независимую оценку")
    values=validate_review(data)
    if db.execute("SELECT 1 FROM pairs WHERE id=?",(pair_id,)).fetchone() is None:
        raise KeyError("Пара не найдена")
    existing=latest_review_rows(db,pair_id)
    if latest_adjudication(db,pair_id):
        raise ValueError("Арбитраж завершён; исправления заблокированы")
    revision=1+max((r["revision"] for r in existing if r["reviewer"]==user["username"]),default=0)
    db.execute("""INSERT INTO reviews(pair_id,reviewer,revision,grade,justification,geography,scale,applicability,geographic_limit,scale_limit,applicability_limit,created_at)
         VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""", (pair_id,user["username"],revision,values["grade"],values["justification"],values["geography"],values["scale"],values["applicability"],values["geographic_limit"],values["scale_limit"],values["applicability_limit"],now()))
    return {"ok":True,"revision":revision,"state":"saved"}


def save_adjudication(db,user:sqlite3.Row,pair_id:int,data:dict) -> dict:
    if user["role"] != "arbiter":
        raise PermissionError("Только арбитр может принимать окончательное решение")
    values=validate_review(data)
    rows=latest_review_rows(db,pair_id)
    if len(rows)!=2:
        raise ValueError("Нужны оценки обоих экспертов")
    if latest_adjudication(db,pair_id):
        raise ValueError("Окончательное решение уже сохранено")
    db.execute("""INSERT INTO adjudications(pair_id,arbiter,revision,grade,justification,geography,scale,applicability,geographic_limit,scale_limit,applicability_limit,review_ids_json,created_at)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",(pair_id,user["username"],1,values["grade"],values["justification"],values["geography"],values["scale"],values["applicability"],values["geographic_limit"],values["scale_limit"],values["applicability_limit"],canon([r["id"] for r in rows]),now()))
    return {"ok":True,"state":"adjudicated"}


def export_reviews(db_path:Path, output:Path, include_predictions:bool=False) -> dict:
    """Operator-only CLI. Do not place output in Git until reviewed for privacy."""
    with managed(db_path) as db:
        users=db.execute("SELECT username,role FROM users ORDER BY username").fetchall()
        if len([u for u in users if u["role"]=="reviewer"])!=2:
            raise ValueError("Ожидалось два reviewer")
        entries=[];final_count=disagreements=awaiting=0
        for pair in db.execute("SELECT * FROM pairs ORDER BY dataset,event_id,company_id"):
            rows=latest_review_rows(db,pair["id"])
            adj=latest_adjudication(db,pair["id"])
            if len(rows)<2: outcome="awaiting_reviews"; awaiting+=1;final=None
            elif adj: outcome="adjudicated";final=reviewed_view(adj); final_count+=1
            elif agreement(*rows): outcome="agreed";final=reviewed_view(rows[0]); final_count+=1
            else: outcome="disagreement";disagreements+=1;final=None
            item={"dataset":pair["dataset"],"dataset_version":pair["dataset_version"],"fixture_sha256":pair["fixture_hash"],
                  "event_id":pair["event_id"],"company_id":pair["company_id"],
                  "review_status":outcome,"reviews":[reviewed_view(r) for r in rows],
                  "adjudication":reviewed_view(adj) if adj else None,"gold":final}
            if include_predictions:
                item["original_model_annotation_not_gold"]=json.loads(pair["original_prediction_json"])
            entries.append(item)
        audit_reviews=[{k:r[k] for k in r.keys()} for r in db.execute("SELECT * FROM reviews ORDER BY id")]
        audit_adjudications=[{k:r[k] for k in r.keys()} for r in db.execute("SELECT * FROM adjudications ORDER BY id")]
        version={r["key"]:r["value"] for r in db.execute("SELECT key,value FROM metadata ORDER BY key")}
    result={"schema_version":"alphich-human-quality-v1","exported_at":now(),"metadata":version,
            "source_fixture_grade_is_not_human_gold":True,"complete":final_count==len(entries),
            "summary":{"total":len(entries),"final":final_count,"awaiting_reviews":awaiting,"disagreements":disagreements},
            "items":entries,"history":{"reviews":audit_reviews,"adjudications":audit_adjudications}}
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_suffix(output.suffix+".tmp")
    if output.exists() or tmp.exists():
        raise ValueError("Выходной файл существует. Не перезаписывайте архив экспертизы.")
    tmp.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    tmp.chmod(0o600)
    tmp.replace(output)
    return result["summary"]


class ReviewHandler(BaseHTTPRequestHandler):
    db_path: Path
    def log_message(self,fmt,*args):
        # Never log authorization header or body.
        sys.stderr.write("review-ui "+fmt%args+"\n")

    def headers_common(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; connect-src 'self'; img-src 'none'; frame-ancestors 'none'; base-uri 'none'")

    def answer(self,code:int,value:dict):
        raw=json.dumps(value,ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.headers_common()
        self.send_header("Content-Type","application/json; charset=utf-8")
        self.send_header("Content-Length",str(len(raw)))
        self.end_headers();self.wfile.write(raw)

    def file(self,name:str,kind:str):
        body=(HERE/name).read_bytes()
        self.send_response(200)
        self.headers_common()
        self.send_header("Content-Type",kind)
        self.send_header("Content-Length",str(len(body)))
        self.end_headers();self.wfile.write(body)

    def do_GET(self):
        url=urlsplit(self.path)
        if url.path in ("/", "/quality-review.html"):
            return self.file("quality_review_ui.html","text/html; charset=utf-8")
        if url.path=="/quality-review.js":
            return self.file("quality_review_ui.js","text/javascript; charset=utf-8")
        if url.path=="/api/health":
            return self.answer(200,{"status":"ok","local_only":True})
        try:
            with managed(self.db_path) as db:
                user=authenticated(db,self.headers.get("X-Review-Token",""))
                if not user: return self.answer(401,{"error":"invalid_review_token"})
                if url.path=="/api/cases": return self.answer(200,summary(db,user))
                if url.path=="/api/case":
                    ids=parse_qs(url.query).get("id",[])
                    if len(ids)!=1 or not ids[0].isdigit(): raise ValueError("Некорректный идентификатор")
                    return self.answer(200,case_view(db,user,int(ids[0])))
                return self.answer(404,{"error":"not_found"})
        except (ValueError,KeyError) as e:
            return self.answer(400,{"error":str(e)})

    def do_POST(self):
        if self.path not in ("/api/review","/api/adjudicate"):
            return self.answer(404,{"error":"not_found"})
        if self.headers.get("Content-Type","").split(";")[0].strip()!="application/json":
            return self.answer(415,{"error":"json_required"})
        try:
            length=int(self.headers.get("Content-Length","0"))
            if not 0<length<=14000: raise ValueError("Недопустимый размер запроса")
            data=json.loads(self.rfile.read(length))
            if not isinstance(data,dict) or type(data.get("id")) is not int or data["id"]<=0:
                raise ValueError("Некорректный id")
            with DB_LOCK,managed(self.db_path) as db:
                db.execute("BEGIN IMMEDIATE")
                user=authenticated(db,self.headers.get("X-Review-Token",""))
                if not user: return self.answer(401,{"error":"invalid_review_token"})
                result=save_review(db,user,data["id"],data) if self.path=="/api/review" else save_adjudication(db,user,data["id"],data)
                db.commit()
            return self.answer(200,result)
        except (ValueError,KeyError,sqlite3.Error) as e:
            return self.answer(400,{"error":str(e)})
        except PermissionError as e:
            return self.answer(403,{"error":str(e)})


def main():
    parser=argparse.ArgumentParser(description="Локальная слепая разметка Alphich (без API/LLM)")
    sub=parser.add_subparsers(dest="command",required=True)
    init=sub.add_parser("init",help="создать БД и три локальных учётных токена")
    init.add_argument("--db",type=Path,default=Path("data/private/quality-review.sqlite3"))
    init.add_argument("--fixtures-dir",type=Path,default=Path("knowledge/fixtures"))
    init.add_argument("--reviewers",nargs=2,required=True,metavar=("EXPERT_A","EXPERT_B"))
    init.add_argument("--arbiter",required=True)
    init.add_argument("--articles",type=Path,help="закрытый JSON: event_id -> {text,published_at}")
    init.add_argument("--contexts",type=Path,help="закрытый JSON: company_id -> {features,verified_at}")
    serve=sub.add_parser("serve",help="локальный веб-интерфейс")
    serve.add_argument("--db",type=Path,default=Path("data/private/quality-review.sqlite3"))
    serve.add_argument("--port",type=int,default=8765)
    enrich=sub.add_parser("enrich",help="внести независимо подтверждённые тексты и профили до начала разметки")
    enrich.add_argument("--db",type=Path,default=Path("data/private/quality-review.sqlite3"))
    enrich.add_argument("--articles",type=Path)
    enrich.add_argument("--contexts",type=Path)
    export=sub.add_parser("export",help="выгрузить оценки и историю, без платного API")
    export.add_argument("--db",type=Path,default=Path("data/private/quality-review.sqlite3"))
    export.add_argument("--output",type=Path,required=True)
    export.add_argument("--include-predictions",action="store_true",help="добавить исходную оценку AI в итоговый файл; только после экспертизы")
    args=parser.parse_args()
    try:
        if args.command=="init":
            result=initialize(args.db,args.fixtures_dir,args.reviewers,args.arbiter,args.articles,args.contexts)
            print("PAIRS_IMPORTED:",result["pairs"])
            print("DB:",result["db"])
            print("ТОКЕНЫ: сохраните локально, НЕ добавляйте в Git/чат!")
            for name,token in result["users"].items():print(f"  {name}: {token}")
        elif args.command=="serve":
            if not args.db.is_file():raise ValueError("Создайте БД командой init")
            class Handler(ReviewHandler): pass
            Handler.db_path=args.db
            server=ThreadingHTTPServer(("127.0.0.1",args.port),Handler)
            print(f"LOCAL_REVIEW_UI: http://127.0.0.1:{args.port}/",flush=True)
            print("Остановить: Ctrl+C",flush=True)
            try:server.serve_forever()
            except KeyboardInterrupt:pass
            finally:server.server_close()
        elif args.command=="enrich":
            with managed(args.db) as db:
                a,c=enrich_data(db,args.articles,args.contexts)
                print(f"ENRICHED: {a} записей статей, {c} контекстов пар")
        elif args.command=="export":
            stats=export_reviews(args.db,args.output,args.include_predictions)
            print("EXPORTED:",args.output)
            print("STATS:",canon(stats))
    except (ValueError,OSError,sqlite3.Error) as e:
        parser.exit(2,f"ERROR: {e}\n")


if __name__=="__main__":main()
