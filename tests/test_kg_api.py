from __future__ import annotations

from fastapi.testclient import TestClient

from ticketflow.api import create_app


def test_kg_api_reports_disabled_fallback(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "disabled"},
    )
    with TestClient(app) as client:
        response = client.get("/api/v1/kg/health")

    assert response.status_code == 200
    assert response.json()["status"] == "disabled"


def test_kg_api_rebuilds_and_queries_ticket_graph_with_memory_backend(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory"},
    )
    with TestClient(app) as client:
        ticket = client.get("/api/v1/tickets", params={"limit": 1}).json()["tickets"][0]

        rebuild = client.post(f"/api/v1/kg/tickets/{ticket['ticket_id']}/rebuild")
        graph = client.get(f"/api/v1/kg/tickets/{ticket['ticket_id']}")
        search = client.get("/api/v1/kg/search", params={"q": ticket["ticket_id"]})

    assert rebuild.status_code == 200
    assert rebuild.json()["status"] in {"ok", "updated"}
    assert graph.status_code == 200
    assert graph.json()["ticket_id"] == ticket["ticket_id"]
    assert any(node["id"] == f"ticket:{ticket['ticket_id']}" for node in graph.json()["nodes"])
    assert search.status_code == 200
    assert search.json()["count"] >= 1
