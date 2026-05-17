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

    response = client.post(f"/api/v1/tickets/{ticket_id}/run", params={"sync": "true"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["ticket_id"] == ticket_id
    assert "thread_id" in payload
    assert "state" in payload


def test_run_ticket_endpoint_creates_async_task_by_default(ticketflow_project):
    client = next(_client(ticketflow_project))
    ticket_id = client.get("/api/v1/tickets", params={"limit": 1}).json()["tickets"][0]["ticket_id"]

    response = client.post(f"/api/v1/tickets/{ticket_id}/run")

    assert response.status_code == 202
    payload = response.json()
    assert payload["ticket_id"] == ticket_id
    assert payload["task_id"]
    assert payload["status"] in {"queued", "running", "succeeded"}

    loaded = client.get(f"/api/v1/tasks/{payload['task_id']}")
    assert loaded.status_code == 200
    assert loaded.json()["task"]["task_id"] == payload["task_id"]


def test_task_cancel_endpoint_handles_unknown_and_existing_tasks(ticketflow_project):
    client = next(_client(ticketflow_project))
    runner = client.app.state.runner or client.app.state.runner
    if runner is None:
        client.get("/readyz")
        runner = client.app.state.runner
    task = runner.repository.create_workflow_task(ticket_id="TCK-CANCEL", mode="async")

    missing = client.post("/api/v1/tasks/missing-task/cancel")
    cancelled = client.post(f"/api/v1/tasks/{task['task_id']}/cancel")

    assert missing.status_code == 404
    assert cancelled.status_code == 200
    assert cancelled.json()["task"]["status"] == "cancelled"


def test_approval_and_outbox_api_surfaces_persistent_state(ticketflow_project):
    client = next(_client(ticketflow_project))
    client.get("/readyz")
    runner = client.app.state.runner
    approval = runner.repository.create_approval_request(
        ticket_id="TCK-APPROVAL",
        thread_id="thread-approval",
        tool_name="issue_refund_request",
        tool_args={"order_id": "ORDER-001"},
        payload={"route_family": "refund_candidate"},
        requested_by="test",
    )
    outbox = runner.repository.create_outbox_event(
        ticket_id="TCK-APPROVAL",
        operation_type="incident_email",
        business_key="incident:TCK-APPROVAL",
        payload={"recipient": "ops@example.com"},
    )

    approvals = client.get("/api/v1/approvals")
    decision = client.post(
        f"/api/v1/approvals/{approval['approval_id']}/decision",
        json={"decision": "approve", "reviewer": "lead", "comment": "同意。"},
    )
    outbox_response = client.get("/api/v1/outbox")

    assert approvals.status_code == 200
    assert approvals.json()["approvals"][0]["approval_id"] == approval["approval_id"]
    assert decision.status_code == 200
    assert decision.json()["approval"]["status"] == "approved"
    assert outbox_response.status_code == 200
    assert outbox_response.json()["events"][0]["event_id"] == outbox["event_id"]
