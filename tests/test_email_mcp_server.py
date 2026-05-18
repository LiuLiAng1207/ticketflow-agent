from __future__ import annotations

from ticketflow import email_mcp_server


def test_incident_email_content_is_utf8_chinese(monkeypatch):
    captured: dict[str, object] = {}

    def fake_send(subject, text_body, html_body, recipients):
        captured.update(
            {
                "subject": subject,
                "text_body": text_body,
                "html_body": html_body,
                "recipients": recipients,
            }
        )
        return {"status": "sent", "delivery_id": "mail-test", "recipients": recipients}

    monkeypatch.setattr(email_mcp_server, "_send_email", fake_send)

    result = email_mcp_server.send_incident_email(
        ticket_id="TCK-001",
        category="technical_issue",
        priority="urgent",
        sla_risk=True,
        customer_tier="enterprise",
        action_type="escalation",
        summary="生产环境不可用",
        evidence_refs=["policy:POL-001"],
        recipients=["ops@example.com"],
    )

    assert result["status"] == "sent"
    assert captured["subject"] == "[TicketFlow 升级通知] TCK-001 | technical_issue | urgent"
    assert "工单 ID：TCK-001" in captured["text_body"]
    assert "高风险工单升级通知" in captured["html_body"]
    assert "policy:POL-001" in captured["html_body"]
    assert "���" not in captured["text_body"]
    assert "锛" not in captured["text_body"]


def test_kb_candidate_email_content_is_utf8_chinese(monkeypatch):
    captured: dict[str, object] = {}

    def fake_send(subject, text_body, html_body, recipients):
        captured.update(
            {
                "subject": subject,
                "text_body": text_body,
                "html_body": html_body,
                "recipients": recipients,
            }
        )
        return {"status": "sent", "delivery_id": "mail-test", "recipients": recipients}

    monkeypatch.setattr(email_mcp_server, "_send_email", fake_send)

    result = email_mcp_server.submit_kb_candidate_email(
        ticket_id="TCK-002",
        category="account_access",
        issue_summary="用户无法登录",
        resolution_summary="建议补充 MFA 排查流程",
        knowledge_gap_reason="现有知识库缺少 MFA 场景",
        suggested_kb_title="MFA 登录排查指南",
        evidence_refs=["history:HIS-001"],
        recipients=["kb@example.com"],
    )

    assert result["status"] == "sent"
    assert captured["subject"] == "[TicketFlow 知识候选] TCK-002 | MFA 登录排查指南"
    assert "知识候选条目" in captured["html_body"]
    assert "知识缺口原因：现有知识库缺少 MFA 场景" in captured["text_body"]
    assert "���" not in captured["text_body"]
    assert "锛" not in captured["text_body"]

