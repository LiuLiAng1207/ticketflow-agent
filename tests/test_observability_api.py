from __future__ import annotations

from fastapi.testclient import TestClient

from ticketflow.api import create_app


def _client(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "OBSERVABILITY_ENABLED": "true"},
    )
    with TestClient(app) as client:
        yield client


def test_api_responses_include_trace_id_and_record_observability_event(ticketflow_project):
    client = next(_client(ticketflow_project))

    response = client.get("/healthz")

    assert response.status_code == 200
    trace_id = response.headers["X-Trace-Id"]
    assert trace_id.startswith("trace-")

    trace = client.get(f"/api/v1/observability/traces/{trace_id}")
    assert trace.status_code == 200
    events = trace.json()["events"]
    assert events
    assert events[0]["trace_id"] == trace_id
    assert events[0]["component"] == "api"
    assert events[0]["span_name"] == "GET /healthz"
    assert events[0]["status"] == "ok"


def test_api_accepts_incoming_trace_id_and_exposes_prometheus_metrics(ticketflow_project):
    client = next(_client(ticketflow_project))

    response = client.get("/readyz", headers={"X-Trace-Id": "trace-client-123"})
    metrics = client.get("/metrics")

    assert response.headers["X-Trace-Id"] == "trace-client-123"
    assert metrics.status_code == 200
    assert "ticketflow_api_requests_total" in metrics.text
    assert "ticketflow_api_request_latency_seconds" in metrics.text


def test_observability_summary_reports_errors_and_queue_counts(ticketflow_project):
    client = next(_client(ticketflow_project))
    client.get("/readyz")
    runner = client.app.state.runner
    runner.repository.record_observability_event(
        trace_id="trace-error-1",
        span_name="worker.failure",
        component="worker",
        status="error",
        latency_ms=3,
        error_type="worker_error",
        payload_summary={"task_id": "task-x"},
    )

    summary = client.get("/api/v1/observability/summary")

    assert summary.status_code == 200
    body = summary.json()
    assert body["errors_by_type"]["worker_error"] >= 1
    assert "workflow_tasks" in body["queues"]
    assert "outbox" in body["queues"]


def test_unknown_trace_returns_404_without_traceback(ticketflow_project):
    client = next(_client(ticketflow_project))

    response = client.get("/api/v1/observability/traces/trace-missing")

    assert response.status_code == 404
    assert "Traceback" not in response.text
