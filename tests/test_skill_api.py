from __future__ import annotations

from fastapi.testclient import TestClient

from ticketflow.api import create_app


def _client(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory"},
    )
    with TestClient(app) as client:
        yield client


def test_skill_api_reload_list_detail_enable_disable_and_run(ticketflow_project):
    client = next(_client(ticketflow_project))

    reload_response = client.post("/api/v1/skills/reload")
    listing = client.get("/api/v1/skills")
    detail = client.get("/api/v1/skills/ticketflow-ops")
    disable = client.post("/api/v1/skills/ticketflow-ops/disable")
    disabled_run = client.post(
        "/api/v1/skills/ticketflow-ops/run",
        json={"input": {"operation": "list_approvals"}, "actor": "api-test"},
    )
    enable = client.post("/api/v1/skills/ticketflow-ops/enable")
    enabled_run = client.post(
        "/api/v1/skills/ticketflow-ops/run",
        json={"input": {"operation": "list_approvals"}, "actor": "api-test"},
    )
    runs = client.get("/api/v1/skills/runs")

    assert reload_response.status_code == 200
    assert reload_response.json()["loaded_count"] >= 5
    assert listing.status_code == 200
    assert any(skill["skill_id"] == "ticketflow-ops" for skill in listing.json()["skills"])
    assert detail.status_code == 200
    assert detail.json()["skill"]["skill_id"] == "ticketflow-ops"
    assert disable.status_code == 200
    assert disable.json()["skill"]["enabled"] is False
    assert disabled_run.status_code == 200
    assert disabled_run.json()["skill_run"]["status"] == "rejected"
    assert enable.status_code == 200
    assert enable.json()["skill"]["enabled"] is True
    assert enabled_run.status_code == 200
    assert enabled_run.json()["skill_run"]["status"] == "succeeded"
    assert runs.status_code == 200
    assert runs.json()["count"] >= 2


def test_skill_api_unknown_skill_returns_404(ticketflow_project):
    client = next(_client(ticketflow_project))

    missing_detail = client.get("/api/v1/skills/missing-skill")
    missing_run = client.post("/api/v1/skills/missing-skill/run", json={"input": {}, "actor": "api-test"})

    assert missing_detail.status_code == 404
    assert missing_run.status_code == 404
