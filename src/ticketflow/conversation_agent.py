from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import TicketRecord
from .repository import RepositoryProtocol
from .skills import SkillRuntime
from .worker import enqueue_run_ticket_workflow


TICKET_ID_PATTERN = re.compile(r"\b(?:TCK|TICKET)-[A-Z0-9-]+\b", re.IGNORECASE)


@dataclass(slots=True)
class AgentChatContext:
    repository: RepositoryProtocol
    project_root: Path
    runner_overrides: dict[str, str] | None = None
    knowledge_graph_store: Any | None = None
    llm_client: Any | None = None


def _extract_ticket_id(message: str) -> str | None:
    match = TICKET_ID_PATTERN.search(message)
    return match.group(0).upper() if match else None


def _contains_any(text: str, tokens: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(token.lower() in lowered for token in tokens)


def _extract_count(message: str, default: int = 10, maximum: int = 50) -> int:
    match = re.search(r"(\d+)\s*(?:张|条|个|件)?", message)
    if match:
        return max(1, min(int(match.group(1)), maximum))
    chinese_numbers = {
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }
    for token, value in chinese_numbers.items():
        if f"{token}张" in message or f"{token}条" in message or f"{token}个" in message:
            return max(1, min(value, maximum))
    return default


def _extract_field(message: str, names: tuple[str, ...]) -> str | None:
    field_aliases = "|".join(re.escape(name) for name in names)
    next_fields = "|".join(
        re.escape(name)
        for name in (
            "标题",
            "工单标题",
            "正文",
            "描述",
            "内容",
            "客户等级",
            "客户",
            "产品",
            "服务",
            "渠道",
            "订单",
        )
    )
    pattern = rf"(?:{field_aliases})\s*[:：]\s*(.*?)(?=(?:[；;\n]\s*(?:{next_fields})\s*[:：])|$)"
    match = re.search(pattern, message, flags=re.DOTALL)
    if match:
        return match.group(1).strip(" ；;\n")
    return None


def _infer_category(text: str) -> str:
    if _contains_any(text, ("退款", "账单", "发票", "扣款", "付款", "refund", "billing")):
        return "billing_refund"
    if _contains_any(text, ("物流", "配送", "发货", "履约", "delivery")):
        return "delivery_issue"
    if _contains_any(text, ("登录", "密码", "账号", "权限", "mfa", "验证码", "access")):
        return "account_access"
    if _contains_any(text, ("故障", "报错", "不可用", "宕机", "服务异常", "error", "incident")):
        return "technical_issue"
    return "general_inquiry"


def _infer_tier(text: str) -> str:
    if "企业" in text or "enterprise" in text.lower():
        return "enterprise"
    if "高级" in text or "premium" in text.lower():
        return "premium"
    return "standard"


def _ticket_summary(ticket: TicketRecord) -> dict[str, Any]:
    return ticket.model_dump(mode="json")


def _event(
    event_type: str,
    title: str,
    *,
    status: str = "completed",
    summary: str = "",
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "event_type": event_type,
        "title": title,
        "status": status,
        "summary": summary,
        "payload": payload or {},
    }


def _response(
    *,
    intent: str,
    reply: str,
    data: dict[str, Any] | None = None,
    actions: list[dict[str, Any]] | None = None,
    events: list[dict[str, Any]] | None = None,
    model_source: str = "deterministic",
    requires_confirmation: bool = False,
) -> dict[str, Any]:
    if events is None:
        events = [_event("intent_detected", "识别用户意图", summary=f"命中意图：{intent}", payload={"intent": intent})]
    return {
        "intent": intent,
        "reply": reply,
        "data": data or {},
        "actions": actions or [],
        "events": events,
        "model_source": model_source,
        "requires_confirmation": requires_confirmation,
    }


def _ops_summary(repository: RepositoryProtocol) -> dict[str, Any]:
    tickets = repository.list_open_tickets(limit=1000)
    by_category: dict[str, int] = {}
    by_tier: dict[str, int] = {}
    refund_related = 0
    risk_watch = 0
    for ticket in tickets:
        category = ticket.expected_category or "unknown"
        by_category[category] = by_category.get(category, 0) + 1
        by_tier[ticket.customer_tier] = by_tier.get(ticket.customer_tier, 0) + 1
        text = f"{ticket.title}\n{ticket.body}".lower()
        if category == "billing_refund" or "退款" in text or "refund" in text:
            refund_related += 1
        if ticket.customer_tier == "enterprise" or category in {"billing_refund", "technical_issue"}:
            risk_watch += 1
    return {
        "open_tickets": len(tickets),
        "enterprise_tickets": by_tier.get("enterprise", 0),
        "risk_watch_tickets": risk_watch,
        "refund_related_tickets": refund_related,
        "by_category": by_category,
        "by_tier": by_tier,
        "sample_tickets": [_ticket_summary(ticket) for ticket in tickets[:5]],
    }


def _identity_response(context: AgentChatContext) -> dict[str, Any]:
    model_status = "已接入，可用于只读解释和自然语言组织" if context.llm_client is not None else "未配置，当前只使用确定性治理能力"
    reply = (
        "我是 TicketFlow 工单协同 Agent。我的职责不是自由聊天，而是帮你查询工单、创建工单、启动处理流程、"
        "解释证据链、查看审批队列、查看 Outbox 投递状态，并把有价值的处理经验沉淀为知识候选。"
        f"当前 DeepSeek 状态：{model_status}。涉及退款、审批、邮件等写操作时，我不会直接执行，必须继续走审批、Outbox、Skill Runtime 和审计链路。"
    )
    return _response(
        intent="identity",
        reply=reply,
        data={"deepseek_connected": context.llm_client is not None},
        actions=[
            {"type": "list_tickets", "method": "GET", "path": "/api/v1/tickets"},
            {"type": "list_approvals", "method": "GET", "path": "/api/v1/approvals"},
            {"type": "list_outbox", "method": "GET", "path": "/api/v1/outbox"},
        ],
        events=[
            _event("intent_detected", "识别用户意图", summary="用户在询问 Agent 身份和能力。", payload={"intent": "identity"}),
            _event("governance_checked", "确认治理边界", summary="写操作不会由模型直接执行。"),
        ],
    )


def _ops_summary_response(context: AgentChatContext) -> dict[str, Any]:
    summary = _ops_summary(context.repository)
    reply = (
        f"当前开放工单 {summary['open_tickets']} 条，其中企业客户工单 {summary['enterprise_tickets']} 条，"
        f"风险关注工单 {summary['risk_watch_tickets']} 条，退款相关工单 {summary['refund_related_tickets']} 条。"
        "建议优先看企业客户、高风险和退款相关队列；低风险标准处理可以批量进入工作流。"
    )
    return _response(
        intent="ops_summary",
        reply=reply,
        data={"summary": summary},
        actions=[
            {"type": "list_tickets", "method": "GET", "path": "/api/v1/tickets?limit=100"},
            {"type": "list_approvals", "method": "GET", "path": "/api/v1/approvals?status=pending"},
        ],
        events=[
            _event("intent_detected", "识别用户意图", summary="用户在询问待处理工单数量。", payload={"intent": "ops_summary"}),
            _event("data_read", "读取运营统计", summary="从工单仓库聚合开放工单、企业客户、风险关注和退款相关数量。", payload=summary),
        ],
    )


def _llm_freeform_response(message: str, context: AgentChatContext) -> dict[str, Any] | None:
    if context.llm_client is None:
        return None
    summary = _ops_summary(context.repository)
    system_prompt = (
        "你是 TicketFlow 工单协同平台的只读对话助手。"
        "你只能解释平台能力、工单处理流程、运营统计和使用建议。"
        "你不能批准退款、不能绕过审批、不能承诺外部邮件已经发送、不能编造不存在的工单。"
        "如果用户要执行写操作，只能建议调用受治理的 API/Skill/审批/Outbox。"
        "回答用中文，简洁明确。"
    )
    user_prompt = json.dumps(
        {
            "user_message": message,
            "ops_summary": summary,
            "available_capabilities": [
                "查询工单",
                "创建工单",
                "启动工作流",
                "查看审批",
                "查看 Outbox",
                "解释知识图谱证据链",
                "运行 Claw 评测",
            ],
        },
        ensure_ascii=False,
    )
    try:
        reply = str(context.llm_client.chat_text(system_prompt, user_prompt)).strip()
    except Exception as exc:  # noqa: BLE001 - UI fallback must remain available.
        return _response(
            intent="freeform_answer",
            reply=(
                "DeepSeek 当前调用失败，我先退回确定性帮助模式。"
                "你可以带上工单编号查询状态、让我统计待处理数量、查看审批队列或查看 Outbox。"
            ),
            data={"error_summary": str(exc)[:240]},
            events=[
                _event("intent_detected", "识别为开放问题", summary="规则未命中具体业务意图。"),
                _event("llm_call", "调用 DeepSeek", status="failed", summary="云端只读兜底失败，已降级。"),
            ],
            model_source="fallback",
        )
    if not reply:
        return None
    return _response(
        intent="freeform_answer",
        reply=reply,
        data={"summary": summary},
        events=[
            _event("intent_detected", "识别为开放问题", summary="规则未命中具体业务意图，进入只读云端兜底。"),
            _event("llm_call", "调用 DeepSeek", summary="DeepSeek 仅参与自然语言解释，不执行写操作。"),
        ],
        model_source="deepseek",
    )


def handle_agent_chat(message: str, context: AgentChatContext) -> dict[str, Any]:
    normalized = message.strip()
    lowered = normalized.lower()
    ticket_id = _extract_ticket_id(normalized)

    if not normalized:
        return _response(intent="empty", reply="请输入要查询或处理的工单问题。", model_source="fallback")

    if _contains_any(normalized, ("你是谁", "你能做什么", "介绍一下", "help", "帮助")):
        return _identity_response(context)

    if _contains_any(normalized, ("待处理工单", "工单数量", "开放工单", "统计工单", "运营概览", "当前有多少")):
        return _ops_summary_response(context)

    if ticket_id and (
        "skill" in lowered
        or ("explain" in lowered and "evidence" in lowered)
        or _contains_any(normalized, ("技能", "证据链", "知识图谱"))
    ):
        result = SkillRuntime(
            context.repository,
            project_root=context.project_root,
            runner_overrides=context.runner_overrides,
            knowledge_graph_store=context.knowledge_graph_store,
        ).run_skill(
            "ticketflow-kg-memory",
            input_payload={"operation": "explain_ticket", "ticket_id": ticket_id},
            actor="conversation_agent",
            ticket_id=ticket_id,
        )
        return _response(
            intent="explain_ticket_graph",
            reply=(
                f"已通过 Skill Runtime 调用 ticketflow-kg-memory 知识图谱能力，为工单 {ticket_id} 生成证据链解释。"
                "这条解释只用于审计和说明，不会绕过证据充分性、审批或工具治理。"
            ),
            data={"ticket_id": ticket_id, "graph": result.result.get("graph", {}), "skill_run": result.model_dump(mode="json")},
            actions=[{"type": "run_skill", "skill_id": "ticketflow-kg-memory"}],
            events=[
                _event("intent_detected", "识别图谱解释意图", payload={"ticket_id": ticket_id}),
                _event("tool_call", "调用 Skill Runtime", summary="调用 ticketflow-kg-memory 解释证据链。"),
            ],
        )

    if ("batch" in lowered and "low" in lowered and "risk" in lowered) or (
        "批量" in normalized and _contains_any(normalized, ("处理", "运行", "启动", "执行", "跑"))
    ):
        requested_count = _extract_count(normalized)
        ticket_ids = [item.upper() for item in re.findall(TICKET_ID_PATTERN, normalized)]
        result = SkillRuntime(
            context.repository,
            project_root=context.project_root,
            runner_overrides=context.runner_overrides,
            knowledge_graph_store=context.knowledge_graph_store,
        ).run_skill(
            "ticketflow-batch-ops",
            input_payload={
                "operation": "batch_run_low_risk",
                "ticket_ids": ticket_ids,
                "limit": requested_count,
                "selection": "auto_low_risk" if not ticket_ids else "explicit_ticket_ids",
            },
            actor="conversation_agent",
        )
        run_payload = result.model_dump(mode="json")
        run_result = run_payload.get("result", {}) if isinstance(run_payload.get("result"), dict) else {}
        return _response(
            intent="run_skill",
            reply=(
                f"已通过 Skill Runtime 为 {run_result.get('count', 0)} 条低风险工单创建批量处理任务；"
                f"本次请求目标为 {run_result.get('requested_count', requested_count)} 条。"
                f"高风险或证据不足的工单会跳过，仍然保留证据充分性、审批和 Outbox 治理链。"
            ),
            data={"skill_run": run_payload},
            actions=[{"type": "run_skill", "skill_id": "ticketflow-batch-ops"}],
            events=[
                _event("intent_detected", "识别批量处理意图", payload={"requested_count": requested_count}),
                _event(
                    "tool_call",
                    "调用批量 Skill",
                    summary="只允许低风险动作进入批处理，高风险样本跳过。",
                    payload=run_result,
                ),
            ],
        )

    if _contains_any(normalized, ("直接批准", "批准所有", "绕过审批", "直接退款", "强制退款")):
        return _response(
            intent="refuse_unsafe_action",
            reply="我不会在聊天里直接批准退款、绕过审批或执行高风险工具。请进入审批队列逐条查看证据、参数和审批策略后再操作。",
            requires_confirmation=True,
            actions=[{"type": "open_approvals", "method": "GET", "path": "/api/v1/approvals"}],
            events=[
                _event("intent_detected", "识别高风险请求", summary="用户试图直接执行或批准高风险动作。"),
                _event("governance_checked", "拒绝绕过治理", status="blocked", summary="高风险动作必须走审批链。"),
            ],
        )

    if _contains_any(normalized, ("审批", "待审批", "approval")):
        approvals = context.repository.list_approval_requests(status="pending", limit=20)
        return _response(
            intent="list_approvals",
            reply=f"当前共有 {len(approvals)} 条待审批请求。高风险动作仍需要在审批接口或运维台中逐条处理。",
            data={"approvals": approvals, "count": len(approvals)},
            actions=[{"type": "list_approvals", "method": "GET", "path": "/api/v1/approvals?status=pending"}],
            events=[
                _event("intent_detected", "识别审批查询意图"),
                _event("data_read", "读取审批队列", payload={"count": len(approvals)}),
            ],
        )

    if "outbox" in lowered or _contains_any(normalized, ("投递", "邮件")):
        events = context.repository.list_outbox_events(limit=20)
        pending = sum(1 for event in events if event.get("status") == "pending")
        return _response(
            intent="list_outbox",
            reply=f"最近 {len(events)} 条 Outbox 事件中，还有 {pending} 条等待投递。",
            data={"events": events, "count": len(events), "pending_count": pending},
            actions=[{"type": "list_outbox", "method": "GET", "path": "/api/v1/outbox"}],
            events=[
                _event("intent_detected", "识别 Outbox 查询意图"),
                _event("data_read", "读取 Outbox 队列", payload={"count": len(events), "pending_count": pending}),
            ],
        )

    if _contains_any(normalized, ("知识", "沉淀", "候选", "提交", "入库")) and _contains_any(normalized, ("知识", "kb", "候选")):
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
            events=[
                _event("intent_detected", "识别知识沉淀意图"),
                _event("tool_call", "写入 Outbox", summary="知识候选不会直接发送，先进入 Outbox。"),
            ],
        )

    if _contains_any(normalized, ("新建工单", "创建工单", "加一个工单", "新增工单")):
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
            events=[
                _event("intent_detected", "识别创建工单意图"),
                _event("data_write", "创建工单", summary="通过受控 repository 写入工单并记录审计。", payload={"ticket_id": ticket.ticket_id}),
            ],
        )

    if ticket_id and _contains_any(normalized, ("处理", "运行", "启动", "执行")):
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
            events=[
                _event("intent_detected", "识别工作流启动意图", payload={"ticket_id": ticket.ticket_id}),
                _event("tool_call", "创建工作流任务", summary="任务进入 worker 队列，不直接在聊天里执行副作用。", payload={"task_id": current_task["task_id"]}),
            ],
        )

    if ticket_id and _contains_any(normalized, ("解释", "为什么", "证据链", "相似案例", "图谱")):
        if context.knowledge_graph_store is None:
            return _response(
                intent="explain_ticket_graph",
                reply="知识图谱服务尚未接入，本次只能查询基础工单状态，不能生成证据链解释。",
                data={"ticket_id": ticket_id, "graph": {"enabled": False, "nodes": [], "edges": [], "node_count": 0, "edge_count": 0}},
                events=[
                    _event("intent_detected", "识别图谱解释意图", payload={"ticket_id": ticket_id}),
                    _event("kg_lookup", "检查知识图谱", status="skipped", summary="KG 未启用。"),
                ],
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
            events=[
                _event("intent_detected", "识别图谱解释意图", payload={"ticket_id": ticket_id}),
                _event("kg_lookup", "读取知识图谱", payload={"node_count": graph.get("node_count", 0), "edge_count": graph.get("edge_count", 0)}),
            ],
        )

    if ticket_id:
        ticket = context.repository.get_ticket(ticket_id)
        return _response(
            intent="query_ticket",
            reply=f"工单 {ticket.ticket_id} 当前状态为 {ticket.status}，客户等级为 {ticket.customer_tier}，产品为 {ticket.product}。",
            data={"ticket": _ticket_summary(ticket)},
            actions=[{"type": "get_ticket", "method": "GET", "path": f"/api/v1/tickets/{ticket.ticket_id}"}],
            events=[
                _event("intent_detected", "识别工单查询意图", payload={"ticket_id": ticket.ticket_id}),
                _event("data_read", "读取工单详情", payload={"ticket_id": ticket.ticket_id, "status": ticket.status}),
            ],
        )

    llm_result = _llm_freeform_response(normalized, context)
    if llm_result is not None:
        return llm_result

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
        model_source="fallback",
        events=[
            _event("intent_detected", "未命中具体业务意图", status="fallback", summary="返回能力说明和示例工单。"),
        ],
    )
