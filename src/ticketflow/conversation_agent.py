from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import TicketRecord
from .repository import RepositoryProtocol
from .worker import enqueue_run_ticket_workflow


TICKET_ID_PATTERN = re.compile(r"\b(?:TCK|TICKET)-[A-Z0-9-]+\b", re.IGNORECASE)


@dataclass(slots=True)
class AgentChatContext:
    repository: RepositoryProtocol
    project_root: Path
    runner_overrides: dict[str, str] | None = None
    knowledge_graph_store: Any | None = None


def _extract_ticket_id(message: str) -> str | None:
    match = TICKET_ID_PATTERN.search(message)
    return match.group(0).upper() if match else None


def _extract_field(message: str, names: tuple[str, ...]) -> str | None:
    for name in names:
        pattern = rf"{re.escape(name)}\s*[:：]\s*([^；;\n]+)"
        match = re.search(pattern, message)
        if match:
            return match.group(1).strip()
    return None


def _infer_category(text: str) -> str:
    if any(token in text for token in ("退款", "账单", "发票", "扣款", "付款")):
        return "billing_refund"
    if any(token in text for token in ("物流", "配送", "发货", "履约")):
        return "delivery_issue"
    if any(token in text for token in ("登录", "密码", "账号", "权限", "MFA", "验证码")):
        return "account_access"
    if any(token in text for token in ("故障", "报错", "不可用", "宕机", "服务异常")):
        return "technical_issue"
    return "general_inquiry"


def _infer_tier(text: str) -> str:
    if "企业" in text:
        return "enterprise"
    if "高级" in text or "premium" in text.lower():
        return "premium"
    return "standard"


def _ticket_summary(ticket: TicketRecord) -> dict[str, Any]:
    return ticket.model_dump(mode="json")


def _response(
    *,
    intent: str,
    reply: str,
    data: dict[str, Any] | None = None,
    actions: list[dict[str, Any]] | None = None,
    requires_confirmation: bool = False,
) -> dict[str, Any]:
    return {
        "intent": intent,
        "reply": reply,
        "data": data or {},
        "actions": actions or [],
        "requires_confirmation": requires_confirmation,
    }


def handle_agent_chat(message: str, context: AgentChatContext) -> dict[str, Any]:
    normalized = message.strip()
    lowered = normalized.lower()

    if any(token in normalized for token in ("直接批准", "批准所有", "绕过审批", "直接退款")):
        return _response(
            intent="refuse_unsafe_action",
            reply="我不会在聊天里直接批准退款、绕过审批或执行高风险工具。请进入审批队列逐条查看证据、参数和审批策略后再操作。",
            requires_confirmation=True,
            actions=[{"type": "open_approvals", "method": "GET", "path": "/api/v1/approvals"}],
        )

    if "审批" in normalized:
        approvals = context.repository.list_approval_requests(status="pending", limit=20)
        return _response(
            intent="list_approvals",
            reply=f"当前共有 {len(approvals)} 条待审批请求。高风险动作仍需要在审批接口或运维台中逐条处理。",
            data={"approvals": approvals, "count": len(approvals)},
            actions=[{"type": "list_approvals", "method": "GET", "path": "/api/v1/approvals?status=pending"}],
        )

    if "outbox" in lowered or "投递" in normalized or "邮件" in normalized:
        events = context.repository.list_outbox_events(limit=20)
        pending = sum(1 for event in events if event.get("status") == "pending")
        return _response(
            intent="list_outbox",
            reply=f"最近 {len(events)} 条 Outbox 事件中，还有 {pending} 条等待投递。",
            data={"events": events, "count": len(events), "pending_count": pending},
            actions=[{"type": "list_outbox", "method": "GET", "path": "/api/v1/outbox"}],
        )

    if "知识" in normalized and any(token in normalized for token in ("沉淀", "候选", "提交", "入库")):
        ticket_id = _extract_ticket_id(normalized) or "TCK-KB-CHAT"
        event = context.repository.create_outbox_event(
            ticket_id=ticket_id,
            operation_type="kb_candidate_email",
            business_key=f"kb_candidate_chat:{ticket_id}:{abs(hash(normalized))}",
            payload={"message": normalized, "source": "agent_chat"},
        )
        return _response(
            intent="submit_kb_candidate",
            reply="已把这条内容登记为知识候选事件，后续由 Outbox Worker 统一投递，避免重复发送。",
            data={"event": event},
            actions=[{"type": "create_outbox", "event_id": event["event_id"]}],
        )

    if any(token in normalized for token in ("新建工单", "创建工单", "加一个工单", "新增工单")):
        title = _extract_field(normalized, ("标题", "工单标题")) or normalized[:40]
        body = _extract_field(normalized, ("正文", "描述", "内容")) or normalized
        product = _extract_field(normalized, ("产品", "服务")) or "未指定产品"
        tier_text = _extract_field(normalized, ("客户等级", "客户")) or normalized
        ticket = context.repository.create_ticket(
            title=title,
            body=body,
            customer_tier=_infer_tier(tier_text),
            product=product,
            channel="web",
            expected_category=_infer_category(f"{title} {body} {product}"),
        )
        context.repository.save_audit_log(
            ticket.ticket_id,
            actor="conversation_agent",
            event_type="ticket_created_from_chat",
            detail="对话式工单 Agent 创建了新工单。",
            payload={"message": normalized},
        )
        return _response(
            intent="create_ticket",
            reply=f"已创建工单 {ticket.ticket_id}，类别初判为 {ticket.expected_category}，当前状态为 {ticket.status}。",
            data={"ticket": _ticket_summary(ticket)},
            actions=[{"type": "get_ticket", "method": "GET", "path": f"/api/v1/tickets/{ticket.ticket_id}"}],
        )

    ticket_id = _extract_ticket_id(normalized)
    explain_tokens = ("\u89e3\u91ca", "\u4e3a\u4ec0\u4e48", "\u8bc1\u636e\u94fe", "\u76f8\u4f3c\u6848\u4f8b", "\u56fe\u8c31")
    if ticket_id and any(token in normalized for token in explain_tokens):
        if context.knowledge_graph_store is None:
            return _response(
                intent="explain_ticket_graph",
                reply="\u77e5\u8bc6\u56fe\u8c31\u670d\u52a1\u5c1a\u672a\u63a5\u5165\uff0c\u672c\u6b21\u53ea\u80fd\u67e5\u8be2\u57fa\u7840\u5de5\u5355\u72b6\u6001\uff0c\u4e0d\u80fd\u751f\u6210\u8bc1\u636e\u94fe\u89e3\u91ca\u3002",
                data={"ticket_id": ticket_id, "graph": {"enabled": False, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0}},
            )
        graph = context.knowledge_graph_store.get_ticket_graph(ticket_id)
        if graph.get("enabled") and not graph.get("nodes"):
            from .knowledge_graph import build_ticket_graph_from_repository

            built_graph = build_ticket_graph_from_repository(context.repository, ticket_id)
            context.knowledge_graph_store.upsert_ticket_graph(built_graph)
            graph = context.knowledge_graph_store.get_ticket_graph(ticket_id)
        return _response(
            intent="explain_ticket_graph",
            reply=(
                f"\u77e5\u8bc6\u56fe\u8c31\u5df2\u4e3a\u5de5\u5355 {ticket_id} \u6c47\u603b {graph.get('node_count', 0)} \u4e2a\u4e1a\u52a1\u8282\u70b9\u3001"
                f"{graph.get('edge_count', 0)} \u6761\u5173\u7cfb\u3002\u5b83\u7528\u4e8e\u89e3\u91ca\u8bc1\u636e\u94fe\u548c\u5904\u7406\u8def\u5f84\uff0c\u4e0d\u4f1a\u7ed5\u8fc7\u539f\u6709\u5ba1\u6279\u4e0e\u8bc1\u636e\u5145\u5206\u6027\u6cbb\u7406\u3002"
            ),
            data={"ticket_id": ticket_id, "graph": graph},
            actions=[{"type": "get_ticket_graph", "method": "GET", "path": f"/api/v1/kg/tickets/{ticket_id}"}],
        )
    if ticket_id and any(token in normalized for token in ("处理", "运行", "启动", "执行")):
        ticket = context.repository.get_ticket(ticket_id)
        task = context.repository.create_workflow_task(ticket_id=ticket.ticket_id, mode="async")
        celery_task_id = enqueue_run_ticket_workflow(
            task_id=str(task["task_id"]),
            project_root=context.project_root,
            runner_overrides=context.runner_overrides,
        )
        current_task = context.repository.get_workflow_task(str(task["task_id"])) or task
        if current_task["status"] in {"queued", "running"}:
            current_task = context.repository.set_workflow_task_celery_id(str(task["task_id"]), celery_task_id)
        return _response(
            intent="run_ticket",
            reply=f"已为工单 {ticket.ticket_id} 创建异步处理任务 {current_task['task_id']}，后续可在任务队列中查看进度。",
            data={"ticket": _ticket_summary(ticket), "task": current_task},
            actions=[{"type": "get_task", "method": "GET", "path": f"/api/v1/tasks/{current_task['task_id']}"}],
        )

    if ticket_id and any(token in normalized for token in ("解释", "为什么", "证据链", "相似案例", "图谱")):
        if context.knowledge_graph_store is None:
            return _response(
                intent="explain_ticket_graph",
                reply="知识图谱服务尚未接入，本次只能查询基础工单状态，不能生成证据链解释。",
                data={"ticket_id": ticket_id, "graph": {"enabled": False, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0}},
            )
        graph = context.knowledge_graph_store.get_ticket_graph(ticket_id)
        if graph.get("enabled") and not graph.get("nodes"):
            from .knowledge_graph import build_ticket_graph_from_repository

            built_graph = build_ticket_graph_from_repository(context.repository, ticket_id)
            context.knowledge_graph_store.upsert_ticket_graph(built_graph)
            graph = context.knowledge_graph_store.get_ticket_graph(ticket_id)
        return _response(
            intent="explain_ticket_graph",
            reply=(
                f"知识图谱已为工单 {ticket_id} 汇总 {graph.get('node_count', 0)} 个业务节点、"
                f"{graph.get('edge_count', 0)} 条关系。它用于解释证据链和处理路径，不会绕过原有审批与证据充分性治理。"
            ),
            data={"ticket_id": ticket_id, "graph": graph},
            actions=[{"type": "get_ticket_graph", "method": "GET", "path": f"/api/v1/kg/tickets/{ticket_id}"}],
        )

    if ticket_id:
        ticket = context.repository.get_ticket(ticket_id)
        return _response(
            intent="query_ticket",
            reply=f"工单 {ticket.ticket_id} 当前状态为 {ticket.status}，客户等级为 {ticket.customer_tier}，产品为 {ticket.product}。",
            data={"ticket": _ticket_summary(ticket)},
            actions=[{"type": "get_ticket", "method": "GET", "path": f"/api/v1/tickets/{ticket.ticket_id}"}],
        )

    tickets = context.repository.list_open_tickets(limit=5)
    return _response(
        intent="help",
        reply="我可以帮你查询工单、创建工单、启动处理流程、查看审批队列、查看 Outbox 投递状态，或提交知识候选。请带上工单编号会更准确。",
        data={"sample_tickets": [_ticket_summary(ticket) for ticket in tickets]},
        actions=[
            {"type": "list_tickets", "method": "GET", "path": "/api/v1/tickets"},
            {"type": "list_approvals", "method": "GET", "path": "/api/v1/approvals"},
            {"type": "list_outbox", "method": "GET", "path": "/api/v1/outbox"},
        ],
    )
