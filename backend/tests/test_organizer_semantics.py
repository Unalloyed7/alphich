"""Organizer clarifications preserve source uncertainty and sentinel handling."""
from alphich.organizer_semantics import revenue_metadata
from app.core import Company, feature_evidence, profile_payload


def test_placeholders_and_malformed_values_never_establish_revenue():
    for raw in (None, "", "0", "1000", "1 000,00", "1e3", "NaN", "Infinity", "-10", "1e9999"):
        info = revenue_metadata(raw)
        assert info["amount"] is None
        assert info["quality_flags"]
    assert revenue_metadata("1 234 500,00")["amount"] == "1234500"


def test_rubles_and_year_are_known_but_amount_does_not_determine_scope_or_segment():
    info = revenue_metadata("25000000000")
    assert info["currency"] == info["unit"] == "RUB"
    assert info["period"] == "2025"
    assert info["scope"] == "company_or_group_unknown"
    assert info["quality_flags"] == []


def test_raw_placeholder_is_preserved_at_source_but_omitted_from_company_evidence():
    row = {"REVENUE_COMPANY": "1000", "MAINOKVED_CODE": "62.01"}
    payload = profile_payload("sample", row, [2])
    assert row["REVENUE_COMPANY"] == "1000"
    assert payload["revenue_raw"] is None
    assert "revenue_placeholder_1000" in payload["quality_flags"]
    # Reading an already imported pre-clarification record is safe too.
    company = Company(id="sample", profile_version="old", revenue_raw="1000", source_rows=[2],
                      products={}, countries=[], quality_flags=[])
    assert not any(f["name"].startswith("revenue_") for f in feature_evidence(company))
