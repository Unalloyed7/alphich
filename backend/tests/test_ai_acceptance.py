"""The acceptance command's default is free; live mode tested with HTTP simulation."""
from datetime import datetime, timedelta, timezone
import json

import httpx
from sqlalchemy import select

from app.core import Company
from app.intelligence.acceptance import run_acceptance
from app.intelligence.acceptance import resume_question
from app.intelligence.service import company_profile
from test_intelligence import BODY, store, runtime, ScriptedGateway


class AcceptanceGateway(ScriptedGateway):
    """Explicit model verdicts for the two fixtures, not a production grade rule."""

    def __call__(self, request):
        body = json.loads(request.content)
        schema = json.loads(body["messages"][0]["content"].split("schema:\n", 1)[1])
        if schema["title"] != "RelevanceJudgment":
            return super().__call__(request)
        data = json.loads(body["messages"][1]["content"])["untrusted_data"]
        industry = next(feature for feature in data["company_features"] if feature["name"] == "okved_code")
        grade, reason = {
            "41.20": (4, "Ставка влияет на стоимость финансирования строительных проектов заказчиков; возможен перенос запуска и авансов, зависимость заказчиков от кредитов нужно проверить."),
            "62.01": (2, "Для разработчика ПО решение о ставке имеет косвенное значение через бюджеты заказчиков; конкретная зависимость портфеля заказов от кредитования не подтверждена."),
        }[industry["value"]]
        self.options["relevance_grade"] = grade
        response = super().__call__(request)
        payload = response.json()
        judgment = json.loads(payload["choices"][0]["message"]["content"])
        judgment.update(reason=reason, company_feature_ids=[industry["id"]])
        payload["choices"][0]["message"]["content"] = json.dumps(judgment, ensure_ascii=False)
        return httpx.Response(200, json=payload)


def files(store, tmp_path, monkeypatch):
    _, factory, _, _ = store
    with factory() as db:
        profiles = [company_profile(c, "cfo") for c in db.scalars(select(Company))]
    profile_path = tmp_path / "profiles.json"
    profile_path.write_text(json.dumps({"companies": profiles}))
    events_path = tmp_path / "events.json"
    stamp = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    events_path.write_text(json.dumps({"events": [{"id": "frozen-news", "title": "Решение Банка России",
        "published_at": stamp, "sources": [{"url": "https://news.example.org/rate", "text": BODY,
        "published_at": stamp}]}]}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("app.intelligence.acceptance.encoder_for", lambda _: None)
    return profile_path, events_path


def test_preflight_cannot_make_a_generative_call_and_preserves_existing_ledger(store, tmp_path, monkeypatch):
    _, _, _, settings = store
    profiles, events = files(store, tmp_path, monkeypatch)
    report = run_acceptance(settings, profiles, events, "data/processed/free.json", max_articles=1)
    assert report["status"] == "preflight_only_no_generation" and report["generative_calls"] == 0
    assert report["phases"][0]["queued"] == 2


def test_complete_ingestion_automation_card_question_and_report_with_scripted_gateway(store, tmp_path, monkeypatch):
    _, _, _, settings = store
    profiles, events = files(store, tmp_path, monkeypatch)
    gateway = AcceptanceGateway()
    def factory(db, configured):
        return runtime(db, configured, gateway)[0]
    report = run_acceptance(settings, profiles, events, "data/processed/complete.json", live=True,
        max_articles=1, max_cards=1, budget_rub="1", runtime_factory=factory)
    assert report["status"] == "passed" and report["question_status"] == "succeeded"
    assert len(report["cards"]) == 1 and report["cards"][0]["relevance_grade"] == 4
    assert report["cards"][0]["company_id"] == "DEMO_A"
    assert {decision["company_id"]: decision["grade"] for decision in report["decisions"]} == {"DEMO_A": 4, "DEMO_B": 2}
    assert sum(name == "RelevanceJudgment" for name, _ in gateway.seen) == 2
    assert sum(name == "ExtractionDraft" for name, _ in gateway.seen) == 1
    assert report["cost_known_micros"] > 0 and report["unknown_reserved_micros"] == 0
    assert "test-secret-never-expose" not in json.dumps(report)


def test_etl_canonical_product_names_are_bound_to_api_source_features(store, tmp_path, monkeypatch):
    from app.core import make_session_factory
    from alphich_kb.personalization.profiles import normalize_company
    _, _, _, settings = store
    profiles, events = files(store, tmp_path, monkeypatch)
    data = json.loads(profiles.read_text())
    data["companies"][0]["products"] = {"lease": True, "loan": False}
    profiles.write_text(json.dumps(data))
    report = run_acceptance(settings, profiles, events, "data/processed/products.json", max_articles=1)
    engine, factory = make_session_factory("sqlite:///" + report["database"])
    try:
        with factory() as db:
            company = db.get(Company, "DEMO_A")
            normalized = normalize_company(company_profile(company, "cfo"))
            assert normalized["products"]["lease"] is True and normalized["bindings"]["lease"]
    finally:
        engine.dispose()


def test_followup_resume_uses_same_ledger_cap_and_idempotency_key(store, tmp_path, monkeypatch):
    _, _, _, settings = store
    profiles, events = files(store, tmp_path, monkeypatch)
    gateway = ScriptedGateway()
    def factory(db, configured):
        return runtime(db, configured, gateway)[0]
    base = "data/processed/base.json"
    report = run_acceptance(settings, profiles, events, base, live=True, max_articles=1,
        max_cards=1, budget_rub="1", runtime_factory=factory)
    first = resume_question(settings, base, "data/processed/followup.json", "Какие данные ассортимента нужны?", runtime_factory=factory)
    count = len(gateway.seen)
    again = resume_question(settings, base, "data/processed/followup-repeated.json", "Какие данные ассортимента нужны?", runtime_factory=factory)
    assert first["status"] == again["status"] == "passed"
    assert first["database"] == report["database"] and first["budget_limit_rub"] == "1"
    assert len(gateway.seen) == count and first["job_id"] == again["job_id"]
