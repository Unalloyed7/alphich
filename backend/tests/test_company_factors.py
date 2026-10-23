import csv

from fastapi.testclient import TestClient

from alphich.company_factors import organizer_factors, parse_employee_band
from app.api import create_app
from app.core import (Base, Company, PRODUCT_COLUMNS, RawCompanyRow, company_factor_context,
                      import_companies, make_session_factory)
from app.intelligence.service import company_profile


def source_row():
    return {"SYNTHETIC_U": "U_FACTORS", "ORG_TYPE": "КОММ_ОРГ", "OKFS_NAME": "ЧАСТНАЯ СОБСТВЕННОСТЬ",
            "COMPANY_AGE_YEARS_GRADATION": "8-15", "ECO_MARK_LABEL": "Оптимизация", "NEGATIVEMARK": "0.0",
            "MAINOKVED_CODE": "42.99", "MAINOKVED_NAME": "Строительство инженерных сооружений",
            "AMOUNT_WORKERS_FNS_GRADATION": "101-250", "AMOUNT_WORKERS_SPARK_GRADATION": "",
            "CL_TYPE": "CLIENT", **{key: "0.0" for key in PRODUCT_COLUMNS}, "FTS_FLAG": "0",
            "FTS_COUNTRY_NAME": "", "REVENUE_COMPANY": "40000000000", "GROUP_ID": "G_FACTORS",
            "CITY": "Москва", "REGION": "Москва"}


def imported_store(tmp_path):
    engine, factory = make_session_factory(f"sqlite:///{tmp_path / 'factors.db'}")
    Base.metadata.create_all(engine)
    row = source_row()
    path = tmp_path / "organizer.csv"
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row), delimiter=";")
        writer.writeheader()
        writer.writerow(row)
    import_companies(path, factory)
    return engine, factory


def test_shared_factors_preserve_labels_and_do_not_invent_employee_counts():
    row = source_row()
    result = organizer_factors(row)
    assert result["negative_mark_raw"] == "0.0"
    assert result["company_age_band"] == "8-15"
    assert result["business_stage_label"] == "Оптимизация"
    assert result["employees_band"] == "101-250"
    assert result["employees_interval"] == {"lower": 101, "upper": 250, "lower_inclusive": True,
        "upper_inclusive": True, "unit": "people", "period": None, "basis": "source_band"}
    assert "company_age_is_not_business_maturity" in result["factor_limitations"]
    conflict = organizer_factors({**row, "AMOUNT_WORKERS_SPARK_GRADATION": "51-100"})
    assert conflict["employees_fns_band"] == "101-250" and conflict["employees_spark_band"] == "51-100"
    assert conflict["employees_band"] is None and conflict["employees_interval"] is None
    assert conflict["employee_band_status"] == "conflict"
    assert parse_employee_band("более 250")["lower_inclusive"] is False
    assert parse_employee_band("unknown") is None
    assert parse_employee_band("0")["lower"] == 0  # No undocumented numeric missing sentinel.


def test_runtime_and_public_profile_share_all_source_bound_factors(tmp_path, monkeypatch):
    engine, factory = imported_store(tmp_path)
    try:
        with factory() as db:
            profile = company_profile(db.get(Company, "U_FACTORS"), "cfo", db=db)
        assert profile["factor_source_status"] == "verified"
        assert profile["employees_fns_band"] == profile["employees_band"] == "101-250"
        assert profile["organization_type"] == "КОММ_ОРГ"
        assert profile["ownership_form"] == "ЧАСТНАЯ СОБСТВЕННОСТЬ"
        facts = {item["name"]: item for item in profile["features"]}
        for name in ("employees_fns_band", "employees_band", "business_stage_label", "organization_type",
                     "ownership_form", "company_age_band", "negative_mark_raw", "city_registration", "client_type"):
            assert facts[name]["value"] == profile[name]
            assert facts[name]["profile_version"] == profile["profile_version"]
            assert "df_Uid_data_hack.csv:row:2" in facts[name]["source_refs"]
        assert facts["fts_flag"]["value"] is False
        monkeypatch.setenv("DEMO_COMPANY_IDS", "U_FACTORS")
        with TestClient(create_app(factory, engine)) as client:
            token = client.post("/api/v1/demo-session", json={"company_id": "U_FACTORS", "role": "cfo"}).json()["token"]
            response = client.get("/api/v1/companies/U_FACTORS/profile", headers={"X-Demo-Session": token})
            assert response.status_code == 200
            public = response.json()
            assert public["features"] == profile["features"]
            assert public["employee_band_status"] == "single_source"
    finally:
        engine.dispose()


def test_raw_factor_version_owner_and_missing_source_are_not_silently_accepted(tmp_path):
    engine, factory = imported_store(tmp_path)
    try:
        with factory.begin() as db:
            company = db.get(Company, "U_FACTORS")
            source = db.get(RawCompanyRow, 2)
            original = dict(source.content)
            source.content = {**original, "AMOUNT_WORKERS_FNS_GRADATION": "1-10"}
            changed = company_factor_context(company, db)
            assert changed["factor_source_status"] == "mismatch" and changed["employees_band"] is None
            source.content = original
            source.company_id = "OTHER_COMPANY"
            changed = company_factor_context(company, db)
            assert changed["factor_source_status"] == "mismatch" and changed["business_stage_label"] is None
            db.delete(source)
            db.flush()
            manual = company_profile(company, "cfo", db=db)
            assert manual["factor_source_status"] == "mismatch"
            assert manual["employees_band"] is None
            assert manual["revenue_raw"] == "40000000000"
            assert manual["features"]  # Manually populated/old profiles remain operational.
    finally:
        engine.dispose()
