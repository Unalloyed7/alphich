"""Offline OpenAPI compatibility checks for frontend ↔ backend integration.

These are route contract checks only; the backend's functional tests exercise ACL,
worker state transitions, budget/receipt validation, and persistence separately.
"""
from app.api import app


def test_frontend_routes_are_present_in_openapi():
    spec = app.openapi()  # Does not connect to the database or call LLM.
    endpoints = {
        ("/api/v1/health", "get"),
        ("/api/v1/demo-session", "post"),
        ("/api/v1/companies", "get"),
        ("/api/v1/companies/{company_id}/profile", "get"),
        ("/api/v1/publications", "post"),
        ("/api/v1/jobs", "get"),
        ("/api/v1/jobs/{job_id}", "get"),
        ("/api/v1/companies/{company_id}/events", "get"),
        ("/api/v1/events/{event_id}", "get"),
        ("/api/v1/sources/{source_id}", "get"),
        ("/api/v1/llm/status", "get"),
        ("/api/v1/companies/{company_id}/feed", "get"),
        ("/api/v1/companies/{company_id}/insights", "post"),
        ("/api/v1/insights/{insight_id}", "get"),
        ("/api/v1/insights/{insight_id}/feedback", "get"),
        ("/api/v1/insights/{insight_id}/feedback", "put"),
        ("/api/v1/insights/{insight_id}/questions", "post"),
        ("/api/v1/answers/{answer_id}", "get"),
        ("/api/v1/evidence/{evidence_id}", "get"),
        ("/api/v1/runs/{run_id}", "get"),
    }
    for path, method in sorted(endpoints):
        assert method in spec["paths"].get(path, {}), f"Missing HTTP contract: {method.upper()} {path}"


def test_paid_post_routes_require_idempotency_key_and_explicit_submission():
    spec = app.openapi()
    for path in (
        "/api/v1/publications",
        "/api/v1/companies/{company_id}/insights",
        "/api/v1/insights/{insight_id}/questions",
    ):
        operation = spec["paths"][path]["post"]
        header_names = {parameter["name"].lower() for parameter in operation.get("parameters", [])
                        if parameter.get("in") == "header"}
        assert "idempotency-key" in header_names, f"Idempotency-Key missing in {path}"
        assert "x-demo-session" in header_names, f"Session header missing in {path}"
        assert "application/json" in operation.get("requestBody", {}).get("content", {}), path
        assert "202" in operation.get("responses", {}), path


def test_read_analysis_routes_do_not_trigger_paid_jobs():
    spec = app.openapi()
    for path in (
        "/api/v1/llm/status",
        "/api/v1/companies/{company_id}/feed",
        "/api/v1/insights/{insight_id}",
        "/api/v1/evidence/{evidence_id}",
        "/api/v1/answers/{answer_id}",
        "/api/v1/runs/{run_id}",
    ):
        assert "get" in spec["paths"][path]
        assert "post" not in spec["paths"][path]
