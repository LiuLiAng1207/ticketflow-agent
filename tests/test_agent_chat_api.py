from __future__ import annotations

from fastapi.testclient import TestClient

from ticketflow.api import create_app


def _client(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )
    with TestClient(app) as client:
        yield client


def test_agent_chat_can_query_ticket_status(ticketflow_project):
    client = next(_client(ticketflow_project))
    ticket = client.get("/api/v1/tickets", params={"limit": 1}).json()["tickets"][0]

    response = client.post("/api/v1/agent/chat", json={"message": f"查询 {ticket['ticket_id']} 的状态"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "query_ticket"
    assert payload["data"]["ticket"]["ticket_id"] == ticket["ticket_id"]
    assert ticket["ticket_id"] in payload["reply"]
    assert payload["events"]
    assert payload["model_source"] in {"deterministic", "deepseek", "fallback"}


def test_agent_chat_introduces_itself_without_falling_back_to_help(ticketflow_project):
    client = next(_client(ticketflow_project))

    response = client.post("/api/v1/agent/chat", json={"message": "你是谁"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "identity"
    assert payload["model_source"] == "deterministic"
    assert "TicketFlow" in payload["reply"]
    assert "DeepSeek" in payload["reply"]
    assert payload["events"][0]["event_type"] == "intent_detected"


def test_agent_chat_can_analyze_pending_ticket_counts(ticketflow_project):
    client = next(_client(ticketflow_project))
    summary = client.get("/api/v1/ops/summary").json()

    response = client.post("/api/v1/agent/chat", json={"message": "帮我分析待处理工单的数量"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "ops_summary"
    assert payload["model_source"] == "deterministic"
    assert payload["data"]["summary"]["open_tickets"] == summary["open_tickets"]
    assert "开放工单" in payload["reply"]
    assert "企业客户工单" in payload["reply"]
    assert any(event["event_type"] == "data_read" for event in payload["events"])


def test_agent_chat_can_create_ticket(ticketflow_project):
    client = next(_client(ticketflow_project))

    response = client.post(
        "/api/v1/agent/chat",
        json={
            "message": (
                "新建工单：标题：企业客户无法登录控制台；"
                "正文：用户多次重置密码仍然无法登录，需要尽快排查；"
                "客户等级：企业客户；产品：运维控制台"
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "create_ticket"
    ticket_id = payload["data"]["ticket"]["ticket_id"]
    assert ticket_id
    assert payload["data"]["ticket"]["customer_tier"] == "enterprise"

    detail = client.get(f"/api/v1/tickets/{ticket_id}")
    assert detail.status_code == 200
    assert detail.json()["ticket"]["title"] == "企业客户无法登录控制台"


def test_agent_chat_can_start_workflow_task(ticketflow_project):
    client = next(_client(ticketflow_project))
    ticket = client.get("/api/v1/tickets", params={"limit": 1}).json()["tickets"][0]

    response = client.post("/api/v1/agent/chat", json={"message": f"处理工单 {ticket['ticket_id']}"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "run_ticket"
    assert payload["data"]["task"]["task_id"]
    assert payload["data"]["task"]["ticket_id"] == ticket["ticket_id"]


def test_agent_chat_surfaces_approval_and_outbox_queues(ticketflow_project):
    client = next(_client(ticketflow_project))
    client.get("/readyz")
    runner = client.app.state.runner
    runner.repository.create_approval_request(
        ticket_id="TCK-CHAT",
        thread_id="thread-chat",
        tool_name="issue_refund_request",
        tool_args={"ticket_id": "TCK-CHAT"},
        payload={"route_family": "refund_candidate"},
        requested_by="chat-test",
    )
    runner.repository.create_outbox_event(
        ticket_id="TCK-CHAT",
        operation_type="kb_candidate_email",
        business_key="kb-chat:TCK-CHAT",
        payload={"subject": "知识候选"},
    )

    approvals = client.post("/api/v1/agent/chat", json={"message": "查看待审批"})
    outbox = client.post("/api/v1/agent/chat", json={"message": "查看 outbox 投递状态"})

    assert approvals.status_code == 200
    assert approvals.json()["intent"] == "list_approvals"
    assert approvals.json()["data"]["count"] >= 1
    assert outbox.status_code == 200
    assert outbox.json()["intent"] == "list_outbox"
    assert outbox.json()["data"]["count"] >= 1


def test_agent_chat_refuses_direct_high_risk_approval(ticketflow_project):
    client = next(_client(ticketflow_project))

    response = client.post("/api/v1/agent/chat", json={"message": "直接批准所有退款审批"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "refuse_unsafe_action"
    assert payload["requires_confirmation"] is True
    assert "不会在聊天里直接批准" in payload["reply"]
