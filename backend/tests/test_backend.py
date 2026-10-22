import csv

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.api import create_app
from app.core import Base, Company, QuarantinedCompany, RawCompanyRow, import_companies, make_session_factory


@pytest.fixture
def store(tmp_path):
    engine, factory = make_session_factory(f"sqlite:///{tmp_path / 'test.db'}")
    Base.metadata.create_all(engine)
    yield engine, factory
    engine.dispose()


def write_csv(path, rows):
    columns = ["SYNTHETIC_U", "MAINOKVED_CODE", "MAINOKVED_NAME", "REVENUE_COMPANY",
               "CL_PRODUCT_LOA_MARK", "REGION", "FTS_FLAG", "FTS_COUNTRY_NAME", "GROUP_ID"]
    with path.open("w", encoding="utf-8", newline="") as f:
        out = csv.DictWriter(f, fieldnames=columns, delimiter=";")
        out.writeheader()
        out.writerows(rows)


def sample_rows():
    return [
        {"SYNTHETIC_U": "U_A", "MAINOKVED_CODE": "41.20", "MAINOKVED_NAME": "Строительство",
         "REVENUE_COMPANY": "20000000000", "CL_PRODUCT_LOA_MARK": "1.0", "REGION": "Москва", "FTS_FLAG": "1", "FTS_COUNTRY_NAME": "Китай"},
        {"SYNTHETIC_U": "U_B", "MAINOKVED_CODE": "62.01", "MAINOKVED_NAME": "Разработка ПО",
         "REVENUE_COMPANY": "", "CL_PRODUCT_LOA_MARK": "", "REGION": "Москва", "FTS_FLAG": "0", "FTS_COUNTRY_NAME": "Китай"},
    ]


def test_import_preserves_unknowns_and_flags(store, tmp_path):
    _, factory = store
    p = tmp_path / "companies.csv"
    write_csv(p, sample_rows())
    report = import_companies(p, factory)
    assert report == {"raw_rows": 2, "unique_ids": 2, "companies": 2,
                      "quarantined_ids": 0, "missing_id_rows": 0}
    with factory() as db:
        a = db.get(Company, "U_A")
        b = db.get(Company, "U_B")
        assert a.products["CL_PRODUCT_LOA_MARK"] is True
        assert b.products["CL_PRODUCT_LOA_MARK"] is None
        assert b.revenue_raw is None
        assert "fts_country_without_flag" in b.quality_flags
        assert len(db.scalars(select(RawCompanyRow)).all()) == 2
    with pytest.raises(ValueError, match="already imported"):
        import_companies(p, factory)
    assert import_companies(p, factory, replace=True)["companies"] == 2


def test_conflicting_repeated_id_quarantined_but_raw_kept(store, tmp_path):
    _, factory = store
    rows = sample_rows()
    rows.append(dict(rows[0], MAINOKVED_CODE="99.99"))
    rows.append(dict(rows[1]))
    p = tmp_path / "conflict.csv"
    write_csv(p, rows)
    stats = import_companies(p, factory)
    assert stats["companies"] == 1
    assert stats["quarantined_ids"] == 1
    with factory() as db:
        assert db.get(Company, "U_A") is None
        assert db.get(QuarantinedCompany, "U_A").source_rows == [2, 4]
        assert db.get(Company, "U_B").source_rows == [3, 5]
        assert len(db.scalars(select(RawCompanyRow)).all()) == 4


def test_access_denied_for_foreign_company_and_unknown_credentials(store, tmp_path, monkeypatch):
    engine, factory = store
    p = tmp_path / "companies.csv"
    write_csv(p, sample_rows())
    import_companies(p, factory)
    monkeypatch.setenv("DEMO_COMPANY_IDS", "U_A,U_B")
    with TestClient(create_app(factory, engine)) as client:
        assert client.get("/api/v1/health").json()["database"] == "ready"
        assert client.get("/api/v1/companies").status_code == 401
        bad = client.post("/api/v1/demo-session", json={"company_id": "U_C", "role": "cfo"})
        assert bad.status_code == 403
        resp = client.post("/api/v1/demo-session", json={"company_id": "U_A", "role": "cfo"})
        assert resp.status_code == 201
        token = resp.json()["token"]
        headers = {"X-Demo-Session": token}
        assert client.get("/api/v1/companies", headers=headers).status_code == 200
        own = client.get("/api/v1/companies/U_A/profile", headers=headers)
        assert own.status_code == 200
        info = own.json()
        assert info["revenue_unit"] == "RUB"
        assert info["features"][0]["id"].startswith("feature_")
        assert client.get("/api/v1/companies/U_B/profile", headers=headers).status_code == 403
        assert client.get("/api/v1/companies/U_A/profile", headers={"X-Demo-Session": "wrong"}).status_code == 401
        monkeypatch.setenv("DEMO_COMPANY_IDS", "U_B")
        assert client.get("/api/v1/companies/U_A/profile", headers=headers).status_code == 403


def test_import_rejects_missing_columns_without_mutating(store, tmp_path):
    _, factory = store
    p = tmp_path / "bad.csv"
    p.write_text("SYNTHETIC_U;REVENUE_COMPANY\nU_A;5\n")
    with pytest.raises(ValueError, match="Missing required"):
        import_companies(p, factory)
    with factory() as db:
        assert db.get(Company, "U_A") is None
