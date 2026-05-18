from __future__ import annotations

import os
import smtplib
from dataclasses import dataclass
from datetime import datetime
from email.header import Header
from email.message import EmailMessage
from email.utils import formataddr, make_msgid
from typing import Any
from uuid import uuid4

from mcp.server.fastmcp import FastMCP


@dataclass(slots=True)
class SMTPSettings:
    host: str
    port: int
    username: str
    auth_code: str
    use_tls: bool
    from_name: str

    @classmethod
    def from_env(cls) -> "SMTPSettings":
        return cls(
            host=os.environ["SMTP_HOST"],
            port=int(os.getenv("SMTP_PORT", "465")),
            username=os.getenv("SMTP_USERNAME", "ticketflow-local@example.com"),
            auth_code=os.getenv("SMTP_AUTH_CODE", ""),
            use_tls=os.getenv("SMTP_USE_TLS", "true").lower() in {"1", "true", "yes", "on"},
            from_name=os.getenv("EMAIL_FROM_NAME", "TicketFlow 工单协同智能体"),
        )


def _send_email(subject: str, text_body: str, html_body: str | None, recipients: list[str]) -> dict[str, Any]:
    settings = SMTPSettings.from_env()
    if not recipients:
        raise ValueError("至少需要一个收件人。")

    message = EmailMessage()
    message["Subject"] = str(Header(subject, "utf-8"))
    message["From"] = formataddr((settings.from_name, settings.username))
    message["To"] = ", ".join(recipients)
    message["Message-ID"] = make_msgid(domain=settings.username.split("@")[-1])
    message.set_content(text_body, charset="utf-8", cte="base64")
    if html_body:
        message.add_alternative(html_body, subtype="html", charset="utf-8", cte="base64")

    if settings.use_tls and settings.port == 465:
        with smtplib.SMTP_SSL(settings.host, settings.port, timeout=30) as server:
            if settings.auth_code:
                server.login(settings.username, settings.auth_code)
            server.send_message(message)
    else:
        with smtplib.SMTP(settings.host, settings.port, timeout=30) as server:
            if settings.use_tls:
                server.starttls()
            if settings.auth_code:
                server.login(settings.username, settings.auth_code)
            server.send_message(message)

    return {
        "delivery_id": f"mail-{uuid4().hex[:12]}",
        "provider_message_id": str(message["Message-ID"]),
        "status": "sent",
        "sent_at": datetime.now().isoformat(),
        "recipients": recipients,
    }


def _format_incident_html(payload: dict[str, Any]) -> str:
    evidence = "".join(f"<li>{item}</li>" for item in payload.get("evidence_refs", []))
    return f"""
    <html>
      <body>
        <h3>高风险工单升级通知</h3>
        <p><b>工单 ID：</b>{payload['ticket_id']}</p>
        <p><b>类别：</b>{payload['category']}</p>
        <p><b>优先级：</b>{payload['priority']}</p>
        <p><b>SLA 风险：</b>{payload['sla_risk']}</p>
        <p><b>客户等级：</b>{payload['customer_tier']}</p>
        <p><b>建议动作：</b>{payload['action_type']}</p>
        <p><b>摘要：</b>{payload['summary']}</p>
        <p><b>关键证据：</b></p>
        <ul>{evidence}</ul>
      </body>
    </html>
    """


def _format_kb_html(payload: dict[str, Any]) -> str:
    evidence = "".join(f"<li>{item}</li>" for item in payload.get("evidence_refs", []))
    return f"""
    <html>
      <body>
        <h3>知识候选条目</h3>
        <p><b>工单 ID：</b>{payload['ticket_id']}</p>
        <p><b>分类：</b>{payload['category']}</p>
        <p><b>建议标题：</b>{payload['suggested_kb_title']}</p>
        <p><b>问题摘要：</b>{payload['issue_summary']}</p>
        <p><b>处理摘要：</b>{payload['resolution_summary']}</p>
        <p><b>知识缺口原因：</b>{payload['knowledge_gap_reason']}</p>
        <p><b>参考证据：</b></p>
        <ul>{evidence}</ul>
      </body>
    </html>
    """


mcp = FastMCP("TicketFlow Email MCP", instructions="发送真实邮件通知与知识候选邮件。")


@mcp.tool(description="发送高风险工单升级通知邮件。", structured_output=True)
def send_incident_email(
    ticket_id: str,
    category: str,
    priority: str,
    sla_risk: bool,
    customer_tier: str,
    action_type: str,
    summary: str,
    evidence_refs: list[str],
    recipients: list[str],
) -> dict[str, Any]:
    subject = f"[TicketFlow 升级通知] {ticket_id} | {category} | {priority}"
    text_body = (
        f"工单 ID：{ticket_id}\n"
        f"类别：{category}\n"
        f"优先级：{priority}\n"
        f"SLA 风险：{sla_risk}\n"
        f"客户等级：{customer_tier}\n"
        f"建议动作：{action_type}\n"
        f"摘要：{summary}\n"
        f"关键证据：\n- " + "\n- ".join(evidence_refs)
    )
    payload = {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "sla_risk": sla_risk,
        "customer_tier": customer_tier,
        "action_type": action_type,
        "summary": summary,
        "evidence_refs": evidence_refs,
    }
    return _send_email(subject, text_body, _format_incident_html(payload), recipients)


@mcp.tool(description="提交知识候选条目邮件。", structured_output=True)
def submit_kb_candidate_email(
    ticket_id: str,
    category: str,
    issue_summary: str,
    resolution_summary: str,
    knowledge_gap_reason: str,
    suggested_kb_title: str,
    evidence_refs: list[str],
    recipients: list[str],
) -> dict[str, Any]:
    subject = f"[TicketFlow 知识候选] {ticket_id} | {suggested_kb_title}"
    text_body = (
        f"工单 ID：{ticket_id}\n"
        f"分类：{category}\n"
        f"建议标题：{suggested_kb_title}\n"
        f"问题摘要：{issue_summary}\n"
        f"处理摘要：{resolution_summary}\n"
        f"知识缺口原因：{knowledge_gap_reason}\n"
        f"关键证据：\n- " + "\n- ".join(evidence_refs)
    )
    payload = {
        "ticket_id": ticket_id,
        "category": category,
        "issue_summary": issue_summary,
        "resolution_summary": resolution_summary,
        "knowledge_gap_reason": knowledge_gap_reason,
        "suggested_kb_title": suggested_kb_title,
        "evidence_refs": evidence_refs,
    }
    return _send_email(subject, text_body, _format_kb_html(payload), recipients)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
