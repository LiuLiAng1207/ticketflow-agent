from __future__ import annotations

from fastapi.testclient import TestClient

from ticketflow.api import create_app


def _client(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash"},
    )
    with TestClient(app) as client:
        yield client


def test_health_and_readiness_endpoints(ticketflow_project):
    client = next(_client(ticketflow_project))

    health = client.get("/healthz")
    ready = client.get("/readyz")

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert ready.status_code == 200
    body = ready.json()
    assert body["status"] == "ready"
    assert body["database"]["backend"] == "sqlite"


def test_ticket_api_lists_and_fetches_seed_ticket(ticketflow_project):
    client = next(_client(ticketflow_project))

    listing = client.get("/api/v1/tickets", params={"limit": 3})
    assert listing.status_code == 200
    tickets = listing.json()["tickets"]
    assert tickets

    ticket_id = tickets[0]["ticket_id"]
    detail = client.get(f"/api/v1/tickets/{ticket_id}")
    assert detail.status_code == 200
    assert detail.json()["ticket"]["ticket_id"] == ticket_id


def test_ticket_api_returns_404_for_unknown_ticket(ticketflow_project):
    client = next(_client(ticketflow_project))

    response = client.get("/api/v1/tickets/UNKNOWN-TICKET")

    assert response.status_code == 404
    assert "UNKNOWN-TICKET" in response.json()["detail"]


def test_ops_summary_and_audit_endpoints(ticketflow_project):
    client = next(_client(ticketflow_project))
    ticket_id = client.get("/api/v1/tickets", params={"limit": 1}).json()["tickets"][0]["ticket_id"]

    summary = client.get("/api/v1/ops/summary")
    audit = client.get(f"/api/v1/tickets/{ticket_id}/audit")

    assert summary.status_code == 200
    assert summary.json()["open_tickets"] >= 1
    assert audit.status_code == 200
    assert "events" in audit.json()


def test_run_ticket_endpoint_reuses_existing_workflow(ticketflow_project):
    client = next(_client(ticketflow_project))
    ticket_id = client.get("/api/v1/tickets", params={"limit": 1}).json()["tickets"][0]["ticket_id"]

    response = client.post(f"/api/v1/tickets/{ticket_id}/run")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticket_id"] == ticket_id
    assert "thread_id" in payload
    assert "state" in payload
