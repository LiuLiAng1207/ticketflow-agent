from __future__ import annotations

from fastapi.testclient import TestClient

from ticketflow.api import create_app
from ticketflow.worker import TASK_REGISTRY, build_knowledge_graph_now


def test_worker_builds_knowledge_graph_with_memory_backend(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory"},
    )
    with TestClient(app) as client:
        ticket = client.get("/api/v1/tickets", params={"limit": 1}).json()["tickets"][0]

    assert "build_knowledge_graph" in TASK_REGISTRY

    result = build_knowledge_graph_now(
        ticket_id=ticket["ticket_id"],
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory"},
    )

    assert result["status"] in {"ok", "updated"}
    assert result["ticket_id"] == ticket["ticket_id"]
    assert result["graph"]["node_count"] >= 1


def test_agent_chat_can_explain_ticket_with_kg(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory"},
    )
    with TestClient(app) as client:
        ticket = client.get("/api/v1/tickets", params={"limit": 1}).json()["tickets"][0]
        client.post(f"/api/v1/kg/tickets/{ticket['ticket_id']}/rebuild")

        response = client.post(
            "/api/v1/agent/chat",
            json={"message": f"解释工单 {ticket['ticket_id']} 为什么这么处理，展示证据链"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "explain_ticket_graph"
    assert payload["data"]["ticket_id"] == ticket["ticket_id"]
    assert "知识图谱" in payload["reply"]
    assert payload["data"]["graph"]["node_count"] >= 1
