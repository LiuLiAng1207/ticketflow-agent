from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
import urllib.error
import urllib.request

import streamlit as st

try:
    from .config import DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL, DEFAULT_OPENAI_COMPAT_CHAT_PATH
    from .graph import TicketFlowRunner
    from .models import ActionProposal, TicketRecord
    from .runtime_control import runtime_report, start_all, stop_all
except ImportError:
    from ticketflow.config import DEFAULT_DEEPSEEK_BASE_URL, DEFAULT_DEEPSEEK_MODEL, DEFAULT_OPENAI_COMPAT_CHAT_PATH
    from ticketflow.graph import TicketFlowRunner
    from ticketflow.models import ActionProposal, TicketRecord
    from ticketflow.runtime_control import runtime_report, start_all, stop_all


CATEGORY_LABELS = {
    "billing_refund": "退款/账单",
    "delivery_issue": "物流履约",
    "technical_issue": "技术故障",
    "account_access": "账号访问",
    "general_inquiry": "一般咨询",
}

PRIORITY_LABELS = {
    "low": "低",
    "medium": "中",
    "high": "高",
    "urgent": "紧急",
}

URGENCY_LABELS = {
    "standard": "标准处理",
    "next_business_day": "下个工作日处理",
    "same_day": "当天处理",
    "sev1": "一级紧急处理",
}

ACTION_LABELS = {
    "refund": "提交退款",
    "escalation": "升级处理",
    "request_info": "补充信息",
    "status_update": "状态更新",
    "troubleshoot": "故障排查",
    "sla_override": "SLA 特批",
    "close_ticket_without_contact": "静默关单",
}

STATUS_LABELS = {
    "open": "待处理",
    "in_progress": "处理中",
    "investigating": "排查中",
    "monitoring": "持续跟踪",
    "waiting_on_customer": "等待客户补充",
    "pending_finance": "等待财务",
    "pending_human": "转人工处理",
    "escalated": "已升级",
}

APPROVAL_STATE_LABELS = {
    "not_required": "无需审批",
    "pending_review": "待审批",
    "approved": "已审批",
    "rejected": "已驳回",
}

CHANNEL_LABELS = {
    "email": "邮件",
    "chat": "在线聊天",
    "web": "网页表单",
}

TIER_LABELS = {
    "standard": "标准版",
    "premium": "高级版",
    "enterprise": "企业版",
}

CUSTOMER_TIER_LABELS = {
    "standard": "标准客户",
    "premium": "高级版客户",
    "enterprise": "企业客户",
}

SOURCE_LABELS = {
    "kb": "知识条目",
    "policy": "策略条款",
    "history": "历史案例",
    "customer": "客户画像",
    "order": "订单信息",
    "template": "回复模板",
    "attachment": "附件证据",
}

ATTACHMENT_EVIDENCE_TYPE_LABELS = {
    "payment_screenshot": "付款截图",
    "error_screenshot": "错误截图",
    "document_form": "表单文档",
    "log_excerpt": "日志片段",
    "generic_attachment": "通用附件",
}

ACTOR_LABELS = {
    "supervisor": "流程编排服务",
    "triage_agent": "分诊服务",
    "knowledge_agent": "知识检索服务",
    "resolution_agent": "处置决策服务",
    "manager": "审批处理",
    "attachment_parser": "附件解析服务",
}

STEP_LABELS = {
    "intake": "受理工单",
    "parse_attachments": "解析工单附件",
    "triage": "工单分诊",
    "retrieve_context": "拉取上下文",
    "assess_sufficiency": "评估上下文充分性",
    "route_action": "动作路由",
    "propose_action": "生成动作建议",
    "approval_gate": "审批处理",
    "execute_action": "执行动作",
    "draft_reply": "生成回复",
    "fact_check_reply": "回复事实校验",
    "finalize": "完结归档",
}

DECISION_LABELS = {
    "approve": "批准执行",
    "edit": "修改后执行",
    "reject": "驳回转人工",
}

ROUTE_FAMILY_LABELS = {
    "refund_candidate": "退款候选路线",
    "escalation_candidate": "升级候选路线",
    "standard_resolution": "标准处理路线",
}

TOOL_LABELS = {
    "issue_refund_request": "提交退款申请工具",
    "create_escalation": "创建升级单工具",
    "update_ticket_status": "更新工单状态工具",
    "mcp.send_incident_email": "发送升级通知邮件",
    "mcp.submit_kb_candidate_email": "提交知识候选邮件",
}

APPROVAL_MODE_LABELS = {
    "approve_reject": "批准/驳回",
    "approve_edit_reject": "批准/修改/驳回",
    "auto": "自动执行",
}

EXECUTION_STATUS_LABELS = {
    "executed": "已执行",
    "needs_handoff": "需要人工接管",
    "rejected": "已驳回",
    "skipped": "已跳过",
    "sent": "已发送",
    "failed": "失败",
}

SOURCE_FIELD_LABELS = {
    "policy": "策略条款",
    "order": "订单信息",
    "history": "历史案例",
    "kb": "知识库",
    "customer": "客户画像",
    "attachment": "附件证据",
}

RETRIEVAL_MODE_LABELS = {
    "lexical": "关键词",
    "vector": "向量",
    "hybrid": "混合",
    "history_rerank": "历史重排",
}

MODEL_BACKEND_LABELS = {
    "minimind_split": "本地双适配模型模式",
    "minimind_api": "本地小模型模式",
    "cloud_api": "云端模型模式",
    "rule": "规则兜底模式",
}

SCENARIO_CONFIG = {
    "退款审批": {"category": "billing_refund", "missing_order": False, "refundable": True},
    "缺单号兜底": {"category": "billing_refund", "missing_order": True},
    "企业故障升级": {"category": "technical_issue", "tier": "enterprise"},
    "发货延迟处理": {"category": "delivery_issue", "missing_order": False},
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _load_env_overrides(project_root: Path) -> dict[str, str]:
    env_path = project_root / ".env"
    if not env_path.exists():
        return {}
    overrides: dict[str, str] = {}
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key:
            overrides[key] = value.strip().strip('"').strip("'")
    return overrides


def _set_default_if_blank(overrides: dict[str, str], key: str, value: str) -> None:
    if not overrides.get(key):
        overrides[key] = value


def _service_api_base_url(project_root: Path | None = None) -> str:
    root = project_root or _project_root()
    overrides = _load_env_overrides(root)
    explicit_url = (
        os.getenv("TICKETFLOW_API_URL")
        or overrides.get("TICKETFLOW_API_URL")
        or os.getenv("API_BASE_URL")
        or overrides.get("API_BASE_URL")
    )
    if explicit_url:
        return explicit_url.rstrip("/")

    host = os.getenv("API_HOST") or overrides.get("API_HOST") or "127.0.0.1"
    port = os.getenv("API_PORT") or overrides.get("API_PORT") or "8000"
    if host in {"0.0.0.0", "::"}:
        host = "127.0.0.1"
    return f"http://{host}:{port}".rstrip("/")


def _fetch_api_json(base_url: str, path: str, *, timeout: float = 1.5) -> dict[str, Any]:
    request = urllib.request.Request(f"{base_url.rstrip('/')}{path}", headers={"Accept": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _load_control_plane_snapshot(project_root: Path, runner: TicketFlowRunner) -> dict[str, Any]:
    api_base_url = _service_api_base_url(project_root)
    try:
        ready = _fetch_api_json(api_base_url, "/readyz")
        tasks = _fetch_api_json(api_base_url, "/api/v1/tasks?limit=20").get("tasks", [])
        approvals = _fetch_api_json(api_base_url, "/api/v1/approvals?limit=20").get("approvals", [])
        outbox_events = _fetch_api_json(api_base_url, "/api/v1/outbox?limit=20").get("events", [])
        return {
            "source": "api",
            "api_base_url": api_base_url,
            "ready": ready,
            "tasks": tasks,
            "approvals": approvals,
            "outbox_events": outbox_events,
            "error": None,
        }
    except (OSError, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        tasks = runner.repository.list_workflow_tasks(limit=20)
        approvals = runner.repository.list_approval_requests(limit=20)
        outbox_events = runner.repository.list_outbox_events(limit=20)
        return {
            "source": "repository",
            "api_base_url": api_base_url,
            "ready": None,
            "tasks": tasks,
            "approvals": approvals,
            "outbox_events": outbox_events,
            "error": str(exc),
        }


def _runtime_env_overrides(project_root: Path, selected_backend: str | None = None) -> dict[str, str]:
    overrides = _load_env_overrides(project_root)
    backend = selected_backend or overrides.get("MODEL_BACKEND") or "cloud_api"
    overrides["MODEL_BACKEND"] = backend
    if backend == "cloud_api":
        _set_default_if_blank(overrides, "DEEPSEEK_BASE_URL", DEFAULT_DEEPSEEK_BASE_URL)
        _set_default_if_blank(overrides, "DEEPSEEK_MODEL", DEFAULT_DEEPSEEK_MODEL)
        _set_default_if_blank(overrides, "OPENAI_COMPAT_CHAT_PATH", DEFAULT_OPENAI_COMPAT_CHAT_PATH)
    return overrides


def _runner() -> TicketFlowRunner:
    project_root = _project_root()
    env_path = project_root / ".env"
    env_mtime = env_path.stat().st_mtime if env_path.exists() else None
    env_overrides = _load_env_overrides(project_root)
    selected_backend = st.session_state.get("ticketflow_selected_backend") or env_overrides.get("MODEL_BACKEND") or "cloud_api"
    overrides = _runtime_env_overrides(project_root, selected_backend)
    cache_key = (str(project_root), env_mtime, selected_backend)
    if st.session_state.get("ticketflow_runner_cache_key") != cache_key:
        old_runner = st.session_state.get("ticketflow_runner")
        if old_runner is not None:
            old_runner.close()
        st.session_state.ticketflow_runner = TicketFlowRunner.from_project_root(
            project_root,
            overrides=overrides,
        )
        st.session_state.ticketflow_runner_cache_key = cache_key
    return st.session_state.ticketflow_runner


def _label(mapping: dict[str, str], value: Any) -> str:
    if value is None:
        return "-"
    return mapping.get(str(value), str(value))


def _safe_label(mapping: dict[str, str], value: Any, fallback: str) -> str:
    if value is None:
        return fallback
    return mapping.get(str(value), fallback)


def _bool_label(value: Any) -> str:
    return "是" if bool(value) else "否"


def _source_list_label(values: list[str] | tuple[str, ...]) -> str:
    return "、".join(_label(SOURCE_FIELD_LABELS, value) for value in values)


def _triage_summary(triage: Any, ticket: Any | None = None) -> str:
    category = _safe_label(CATEGORY_LABELS, getattr(triage, "category", None), "未识别类别")
    priority = _safe_label(PRIORITY_LABELS, getattr(triage, "priority", None), "未识别")
    urgency = _safe_label(URGENCY_LABELS, getattr(triage, "urgency", None), "按当前优先级处理")
    tier = _safe_label(CUSTOMER_TIER_LABELS, getattr(ticket, "customer_tier", None), "未知等级客户")
    channel = _safe_label(CHANNEL_LABELS, getattr(ticket, "channel", None), "未知渠道")
    sla_sentence = "服务时效风险已触发，需要优先跟进。" if getattr(triage, "sla_risk", False) else "服务时效风险未触发，按当前优先级推进。"

    return (
        f"系统将该工单识别为“{category}”问题。"
        f"客户画像为{tier}，来源渠道为{channel}，优先级为{priority}，处理时效为{urgency}。"
        f"{sla_sentence}"
        "分诊结论来自工单标题、正文、客户等级和渠道等结构化信息，后续会进入证据检索、充分性判断和动作治理链。"
    )


def _action_summary(action: Any, sufficiency: Any | None = None) -> str:
    action_label = _safe_label(ACTION_LABELS, getattr(action, "action_type", None), "受控处理")
    status_label = _safe_label(STATUS_LABELS, getattr(action, "target_status", None), "处理中")
    tool_label = _safe_label(TOOL_LABELS, getattr(action, "suggested_tool", None), "受控工具")
    route_label = _safe_label(ROUTE_FAMILY_LABELS, getattr(action, "route_family", None), "标准处理路线")
    approval_label = "需要人工审批" if getattr(action, "requires_approval", False) else "无需人工审批"

    sufficient = bool(getattr(sufficiency, "sufficient", getattr(action, "sufficiency_passed", True)))
    missing_sources = list(getattr(sufficiency, "missing_sources", getattr(action, "missing_sources", [])) or [])
    if sufficient:
        evidence_sentence = "上下文证据已通过充分性检查，动作被限制在当前治理路线允许的安全范围内。"
    elif missing_sources:
        evidence_sentence = f"上下文证据尚不充分，当前缺少{_source_list_label(missing_sources)}，系统会优先采用保守动作。"
    else:
        evidence_sentence = "上下文证据仍需谨慎使用，系统会优先采用保守动作并保留审计记录。"

    return (
        f"系统建议执行“{action_label}”，目标状态为“{status_label}”，调用工具为“{tool_label}”。"
        f"当前动作路线为“{route_label}”，审批策略为“{approval_label}”。"
        f"{evidence_sentence}"
    )


def _reply_internal_summary(reply: Any, state: dict[str, Any]) -> str:
    status_label = _safe_label(STATUS_LABELS, getattr(reply, "status", None), "处理中")
    action = state.get("proposed_action")
    action_label = _safe_label(ACTION_LABELS, getattr(action, "action_type", None), "受控处理")
    fact_check = state.get("reply_fact_check")
    if fact_check is None:
        fact_sentence = "回复事实校验结果尚未写入当前页面。"
    elif getattr(fact_check, "passed", False):
        ratio = getattr(fact_check, "supported_claim_ratio", None)
        ratio_text = f"，证据支持比例为 {ratio}" if ratio is not None else ""
        fact_sentence = f"回复事实校验已通过{ratio_text}。"
    else:
        fact_sentence = "回复事实校验未通过，系统已触发重写、降级或人工接管策略。"

    return (
        f"客户回复状态为“{status_label}”，对应动作是“{action_label}”。"
        f"{fact_sentence}"
        "该备注由结构化运行状态生成，用于演示和审计摘要；模型原始备注仍可在节点原始数据中查看。"
    )


def _sufficiency_summary(result: Any) -> str:
    route_family = getattr(result, "route_family", "")
    sufficient = bool(getattr(result, "sufficient", False))
    missing_sources = list(getattr(result, "missing_sources", []) or [])
    required_sources = list(getattr(result, "required_sources", []) or [])
    supporting_doc_ids = list(getattr(result, "supporting_doc_ids", []) or [])

    if sufficient:
        if route_family == "refund_candidate":
            return "退款路线证据充分：已同时具备退款策略条款和订单信息，可以进入受约束的退款动作建议，并在执行前进入人工审批。"
        if route_family == "escalation_candidate":
            return "升级路线证据充分：已具备策略条款和可参考的历史案例，可以进入受约束的升级动作建议。"
        return "当前上下文证据满足标准处理路线要求，可以继续生成普通处理建议。"

    missing_label = _source_list_label(missing_sources) if missing_sources else "关键证据"
    required_label = _source_list_label(required_sources) if required_sources else "当前路线所需证据"
    has_order_doc = any("order" in str(doc_id).lower() or "订单信息" in _zh_text(doc_id) for doc_id in supporting_doc_ids)
    has_policy_doc = any("policy" in str(doc_id).lower() or "策略条款" in _zh_text(doc_id) for doc_id in supporting_doc_ids)

    if route_family == "refund_candidate":
        if "policy" in missing_sources and has_order_doc:
            return (
                "订单证据已找到，但缺少可放行退款的明确策略条款。"
                "当前命中的策略内容更偏向限制性或兜底说明，只能支撑保守处理，不能支撑直接提交退款。"
                "因此系统会先请求补充信息或转入人工确认，避免在策略依据不足时执行财务动作。"
            )
        if "order" in missing_sources and has_policy_doc:
            return (
                "策略条款已命中，但缺少可核验的订单信息。"
                "退款属于财务敏感动作，必须先确认订单号、订单状态和是否满足退款条件，不能只凭客户描述直接提交退款。"
            )
        return (
            f"退款路线证据不足：该路线要求 {required_label}，当前缺少 {missing_label}。"
            "系统不会直接提交退款，而会降级为补充信息或人工确认。"
        )

    if route_family == "escalation_candidate":
        if "history" in missing_sources:
            return (
                "升级路线证据不足：虽然当前工单可能有风险，但缺少足够相似、处理结果可靠的历史案例支撑。"
                "系统会先转入人工队列或状态更新，避免模型激进创建升级单。"
            )
        return (
            f"升级路线证据不足：该路线要求 {required_label}，当前缺少 {missing_label}。"
            "系统会先保守转人工或更新状态。"
        )

    return (
        f"当前上下文仍有证据缺口：缺少 {missing_label}。"
        "标准处理路线不会被强制阻断，但后续动作和回复会继续受到治理链约束。"
    )


def _zh_text(value: Any) -> str:
    """Translate common internal enum/debug phrases for presentation only."""
    if value is None:
        return ""
    text = str(value)
    replacements = {
        "Deterministic sufficiency validator found all required evidence.": "确定性证据充分性校验确认：当前路线所需证据已经齐全。",
        "Deterministic sufficiency validator found missing sources: order.": "确定性证据充分性校验发现缺少证据来源：订单信息。",
        "Deterministic sufficiency validator found missing sources: policy.": "确定性证据充分性校验发现缺少证据来源：策略条款。",
        "Deterministic sufficiency validator found missing sources: history.": "确定性证据充分性校验发现缺少证据来源：历史案例。",
        "No LLM client configured for context sufficiency; used deterministic validator.": "上下文充分性节点未连接大模型，已使用确定性校验器兜底。",
        "Deterministic constrained fallback was used because the structured action model failed.": "由于结构化动作模型输出不可用，系统已使用受约束的确定性兜底动作。",
        "Refund route had sufficient evidence, so the workflow fell back to a deterministic refund action after the structured action model failed.": "退款路线证据充分；由于结构化动作模型输出不可用，系统使用确定性退款动作作为安全兜底。",
        "Escalation route had sufficient evidence, so the workflow fell back to a deterministic escalation action after the structured action model failed.": "升级路线证据充分；由于结构化动作模型输出不可用，系统使用确定性升级动作作为安全兜底。",
        "The workflow fell back to deterministic troubleshooting because the structured action model failed under a standard-resolution route.": "标准处理路线下结构化动作模型输出不可用，系统改用确定性故障排查动作兜底。",
        "The workflow fell back to a deterministic status update because the structured action model failed under a standard-resolution route.": "标准处理路线下结构化动作模型输出不可用，系统改用确定性状态更新动作兜底。",
        "The workflow fell back to requesting more information because the structured action model failed and no safer constrained action was available.": "结构化动作模型输出不可用，且没有更安全的受约束动作，因此系统改为请求客户补充信息。",
        "Standard-resolution path stays in constrained action generation without mandatory blocking evidence.": "标准处理路线不设置强阻断证据，但动作生成仍限制在安全动作集合内。",
        "Reply matches the deterministic conservative template for this action route.": "回复与当前动作路线的确定性保守模板一致。",
        "MiniMind 结构化适配器完成工单分诊。": "本地结构化模型已完成工单分诊。",
        "Refund path is missing critical evidence, so the workflow must request more information before taking a financial action.": "退款路径缺少关键证据，因此在执行财务动作前必须先向客户请求补充信息。",
        "Refund path has sufficient evidence, so the workflow enters a route-family constrained refund decision instead of reopening the full action space.": "退款路径证据充分，因此系统进入受路线约束的退款决策，不重新开放全部动作空间。",
        "Escalation path is missing critical evidence, so the workflow must defer to a human queue instead of issuing an aggressive escalation.": "升级路径缺少关键证据，因此系统转入人工队列，而不是激进创建升级单。",
        "Escalation path has sufficient evidence, so the workflow enters a route-family constrained escalation decision instead of allowing lower-risk fallback actions.": "升级路径证据充分，因此系统进入受路线约束的升级处理。",
        "Governance-safe fallback reply bypassed LLM fact-check because it was generated from a deterministic conservative route.": "该回复来自治理链的安全保守路线，因此不再依赖大模型事实校验。",
        "Deterministic fact-check verified that every reply sentence stayed within the allowed supported claims.": "确定性事实校验确认回复内容均落在允许的证据支持范围内。",
        "Deterministic fact-check found that the reply missed route-critical supported claims.": "确定性事实校验发现回复缺少当前路线必须表达的关键信息。",
        "Routine ticket status updates are auto-executed.": "常规工单状态更新默认自动执行。",
        "Refund requests are financially sensitive and require explicit approval with sufficient policy and order evidence.": "退款请求涉及财务风险，必须在策略条款和订单证据充分时由人工明确审批。",
        "Escalations are sensitive operational actions and require explicit approval with sufficient policy and history evidence.": "升级处理属于敏感运营动作，必须在策略条款和历史案例证据充分时由人工明确审批。",
        "KB candidate email is non-blocking and does not require separate approval in V1.": "知识候选邮件在 V1 中不阻塞主流程，也不单独进入审批。",
        "No explicit approval policy configured.": "未配置单独审批策略。",
        "governance_safe_fallback": "治理链安全兜底",
        "Current tool approval policy.": "当前工具审批策略。",
        "used deterministic validator": "已使用确定性校验器",
        "used deterministic constrained fallback": "已使用受约束的确定性兜底",
        "structured action model failed": "结构化动作模型输出不可用",
        "structured action model 失败": "结构化动作模型输出不可用",
        "llm_minimind": "本地 MiniMind",
        "llm_cloud": "云端模型",
        "ValueError": "结构解析错误",
        "Refund tool is missing required execution arguments.": "退款工具缺少必要执行参数。",
        "high-risk": "高风险",
        "lower-risk": "低风险",
        "priority=low": "优先级=低",
        "priority=medium": "优先级=中",
        "priority=high": "优先级=高",
        "priority=urgent": "优先级=紧急",
        "优先级=low": "优先级=低",
        "优先级=medium": "优先级=中",
        "优先级=high": "优先级=高",
        "优先级=urgent": "优先级=紧急",
        "知识 Agent": "知识检索服务",
        "决策 Agent": "处置决策服务",
        "监督编排器": "流程编排服务",
        "policy:": "策略条款：",
        "kb:": "知识库：",
        "history:": "历史案例：",
        "order:": "订单信息：",
        "提交退款_candidate": "退款候选路线",
        "升级处理_candidate": "升级候选路线",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    for raw, label in CATEGORY_LABELS.items():
        text = text.replace(raw, label)
    for raw, label in ACTION_LABELS.items():
        text = text.replace(raw, label)
    for raw, label in STATUS_LABELS.items():
        text = text.replace(raw, label)
    for raw, label in ROUTE_FAMILY_LABELS.items():
        text = text.replace(raw, label)
    for raw, label in TOOL_LABELS.items():
        text = text.replace(raw, label)
    for raw, label in EXECUTION_STATUS_LABELS.items():
        text = text.replace(raw, label)
    for raw, label in SOURCE_FIELD_LABELS.items():
        text = text.replace(raw, label)
    final_replacements = {
        "提交退款_candidate": "退款候选路线",
        "升级处理_candidate": "升级候选路线",
        "standard_resolution": "标准处理路线",
        "issue_提交退款_request": "提交退款申请工具",
        "create_升级处理": "创建升级单工具",
        "update_ticket_状态更新": "更新工单状态工具",
        "本地 本地小模型": "本地小模型",
        "从 本地小模型 回退": "从本地小模型回退",
        "从本地 本地小模型回退": "从本地小模型回退",
        "处置决策服务 提出动作": "处置决策服务提出动作",
        "知识检索服务 已为": "知识检索服务已为",
        "； 已使用": "；已使用",
        "; 已使用": "；已使用",
        ";": "；",
    }
    for src, dst in final_replacements.items():
        text = text.replace(src, dst)
    text = text.replace("SLA 风险", "服务时效风险")
    text = text.replace("SLA 正常", "服务时效正常")
    text = text.replace("SLA 有风险", "存在服务时效风险")
    text = text.replace("SLA", "服务时效")
    text = text.replace("RAG 检索", "知识检索")
    text = text.replace("LLM", "大模型")
    text = text.replace("MiniMind", "本地小模型")
    text = text.replace("route_family=", "动作路线=")
    text = text.replace("sufficient=True", "证据充分=是")
    text = text.replace("sufficient=False", "证据充分=否")
    text = text.replace("passed=True", "通过=是")
    text = text.replace("passed=False", "通过=否")
    text = text.replace("rewritten=True", "已重写=是")
    text = text.replace("rewritten=False", "已重写=否")
    text = text.replace("downgraded=True", "已降级=是")
    text = text.replace("downgraded=False", "已降级=否")
    text = text.replace("status=", "状态=")
    text = text.replace("target_status=", "目标状态=")
    text = text.replace("结果=approve", "结果=批准执行")
    text = text.replace("结果=edit", "结果=修改后执行")
    text = text.replace("结果=reject", "结果=驳回转人工")
    text = text.replace("本地 本地小模型", "本地小模型")
    text = text.replace("从 本地小模型 回退", "从本地小模型回退")
    text = text.replace("提交退款_candidate", "退款候选路线")
    text = text.replace("升级处理_candidate", "升级候选路线")
    text = text.replace("issue_提交退款_request", "提交退款申请工具")
    return text


def _badge(label: str, tone: str = "neutral") -> str:
    return f'<span class="tf-badge tf-badge-{tone}">{_zh_text(label)}</span>'


def _doc_id_label(value: str) -> str:
    return _zh_text(value)


def _source_badge(source: str) -> str:
    mapping = {
        "rule": ("安全兜底", "neutral"),
        "llm_cloud": ("云端模型", "accent"),
        "llm_minimind": ("本地模型", "success"),
        "human_edit": ("人工修订", "warning"),
        "governance": ("治理路线", "warning"),
    }
    label, tone = mapping.get(source, (source, "neutral"))
    return _badge(label, tone)


def _mask_email(email: str | None) -> str:
    if not email:
        return "未配置"
    name, _, domain = email.partition("@")
    if len(name) <= 3:
        masked = name[:1] + "***"
    else:
        masked = name[:3] + "***" + name[-2:]
    return f"{masked}＠{domain}" if domain else masked


def _missing_email_settings(settings: Any) -> list[str]:
    missing: list[str] = []
    if not settings.smtp_host:
        missing.append("发信服务器")
    if not settings.smtp_username:
        missing.append("发件邮箱")
    if not settings.smtp_auth_code:
        missing.append("邮箱授权码")
    return missing


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --tf-bg: #eef3f8;
            --tf-surface: #ffffff;
            --tf-surface-muted: #f8fafc;
            --tf-border: #dbe3ec;
            --tf-border-strong: #c8d3df;
            --tf-text: #14202b;
            --tf-muted: #5f6f82;
            --tf-accent: #0f5b8f;
            --tf-accent-soft: #e6f4fb;
            --tf-ink: #10243a;
            --tf-navy: #12324b;
            --tf-success: #157347;
            --tf-warning: #b76e00;
            --tf-danger: #c92a2a;
        }

        .stApp {
            background: var(--tf-bg);
            color: var(--tf-text);
        }

        #MainMenu, .stDeployButton, [data-testid="stAppDeployButton"] {
            visibility: hidden;
            display: none;
        }

        [data-testid="stHeader"] {
            background: rgba(238, 243, 248, 0.82);
            backdrop-filter: blur(10px);
        }

        header [data-testid="stToolbar"] {
            visibility: visible;
            display: flex;
        }

        [data-testid="stSidebar"] {
            background: #f8fbff;
            border-right: 1px solid #c6d4e0;
            box-shadow: 10px 0 30px rgba(16, 36, 58, 0.08);
        }

        [data-testid="stSidebar"] > div {
            background:
                linear-gradient(180deg, rgba(16, 36, 58, 0.08) 0%, rgba(248, 251, 255, 0) 160px),
                #f8fbff;
            overflow-x: visible;
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--tf-ink);
            letter-spacing: 0.01em;
        }

        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span,
        [data-testid="stSidebar"] small {
            color: var(--tf-text);
        }

        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] span {
            color: var(--tf-muted) !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] {
            display: grid;
            gap: 0.45rem;
            margin: 0.35rem 0 0.65rem;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label {
            width: 100%;
            min-height: 42px;
            padding: 0.55rem 0.65rem;
            border: 1px solid var(--tf-border-strong);
            border-radius: 13px;
            background: #ffffff;
            color: var(--tf-text) !important;
            box-shadow: 0 6px 14px rgba(16, 36, 58, 0.04);
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:hover {
            border-color: #9eb7ca;
            background: #f1f7fb;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
            border-color: #2f7da8;
            background: #e6f4fb;
            color: #0b3b59 !important;
        }

        [data-testid="stSidebar"] div[role="radiogroup"] label p {
            color: inherit !important;
            font-weight: 700;
            line-height: 1.35;
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] {
            border-radius: 13px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            background: #ffffff;
            color: var(--tf-text);
        }

        [data-testid="stSidebar"] [data-testid="stAlert"] * {
            color: var(--tf-text) !important;
        }

        .block-container {
            padding-top: 1.25rem;
            padding-bottom: 1.5rem;
            max-width: 1540px;
        }

        [data-testid="stMetric"] {
            background: var(--tf-surface);
            border: 1px solid var(--tf-border);
            border-radius: 12px;
            padding: 0.8rem 0.9rem;
            box-shadow: none;
        }

        [data-testid="stVerticalBlockBorderWrapper"] {
            background: transparent;
        }

        .tf-page-header {
            position: relative;
            overflow: hidden;
            background:
                radial-gradient(circle at 84% 18%, rgba(58, 169, 189, 0.38), transparent 28%),
                linear-gradient(135deg, #10243a 0%, #163f5f 45%, #1f6f8d 100%);
            border: 1px solid rgba(255, 255, 255, 0.24);
            border-radius: 28px;
            padding: 1.65rem 1.8rem;
            box-shadow: 0 22px 48px rgba(16, 36, 58, 0.18);
            margin-bottom: 1rem;
        }

        .tf-page-header:after {
            content: "";
            position: absolute;
            right: -5rem;
            top: -7rem;
            width: 18rem;
            height: 18rem;
            border-radius: 50%;
            border: 1px solid rgba(255, 255, 255, 0.23);
        }

        .tf-hero-kicker {
            position: relative;
            z-index: 1;
            color: #b7dff0;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }

        .tf-page-title {
            position: relative;
            z-index: 1;
            font-size: 2.15rem;
            font-weight: 800;
            color: #ffffff;
            letter-spacing: -0.03em;
            margin-bottom: 0.2rem;
        }

        .tf-page-subtitle {
            position: relative;
            z-index: 1;
            color: #d8e9f1;
            font-size: 1rem;
            line-height: 1.7;
            max-width: 760px;
        }

        .tf-status-rail {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.85rem 0 1.1rem;
        }

        .tf-health-item {
            background: rgba(255, 255, 255, 0.84);
            border: 1px solid rgba(196, 211, 224, 0.92);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            box-shadow: 0 12px 26px rgba(28, 54, 76, 0.06);
        }

        .tf-health-label {
            color: var(--tf-muted);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
        }

        .tf-health-value {
            color: var(--tf-ink);
            font-size: 1.05rem;
            font-weight: 800;
            margin-top: 0.28rem;
        }

        .tf-health-caption {
            color: var(--tf-muted);
            font-size: 0.78rem;
            line-height: 1.45;
            margin-top: 0.25rem;
        }

        .tf-service-row {
            background: rgba(255, 255, 255, 0.9);
            border: 1px solid var(--tf-border);
            border-radius: 16px;
            padding: 0.75rem 0.8rem;
            margin-bottom: 0.55rem;
        }

        .tf-service-name {
            color: var(--tf-ink);
            font-weight: 800;
            font-size: 0.92rem;
        }

        .tf-service-meta {
            color: var(--tf-muted);
            font-size: 0.76rem;
            line-height: 1.45;
            margin-top: 0.2rem;
        }

        @media (max-width: 980px) {
            .tf-status-rail {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }

        @media (max-width: 640px) {
            .tf-status-rail {
                grid-template-columns: 1fr;
            }
        }

        .tf-panel {
            background: var(--tf-surface);
            border: 1px solid var(--tf-border);
            border-radius: 20px;
            padding: 0.95rem 1rem;
            box-shadow: 0 10px 24px rgba(22, 32, 43, 0.055);
            margin-bottom: 0.75rem;
        }

        .tf-panel-title {
            font-size: 0.95rem;
            font-weight: 700;
            color: var(--tf-text);
            margin-bottom: 0.45rem;
        }

        .tf-caption {
            font-size: 0.9rem;
            color: var(--tf-muted);
            line-height: 1.55;
        }

        .tf-key-value {
            display: grid;
            grid-template-columns: 110px 1fr;
            gap: 0.35rem 0.75rem;
            margin-top: 0.65rem;
            font-size: 0.96rem;
        }

        .tf-key {
            color: var(--tf-muted);
        }

        .tf-value {
            color: var(--tf-text);
            font-weight: 600;
        }

        .tf-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.18rem 0.5rem;
            border-radius: 999px;
            font-size: 0.76rem;
            font-weight: 600;
            margin-right: 0.35rem;
            margin-bottom: 0.25rem;
            border: 1px solid var(--tf-border);
        }

        .tf-badge-neutral {
            background: #f8fafc;
            color: #334155;
        }

        .tf-badge-accent {
            background: var(--tf-accent-soft);
            color: var(--tf-accent);
            border-color: #cddcff;
        }

        .tf-badge-success {
            background: #edf9f2;
            color: var(--tf-success);
            border-color: #c9ecd6;
        }

        .tf-badge-warning {
            background: #fff8eb;
            color: var(--tf-warning);
            border-color: #f3dfad;
        }

        .tf-badge-danger {
            background: #fff1f2;
            color: var(--tf-danger);
            border-color: #f1c7cb;
        }

        .tf-section-label {
            font-size: 0.86rem;
            font-weight: 700;
            color: #334155;
            margin: 0.15rem 0 0.55rem 0;
            text-transform: uppercase;
            letter-spacing: 0.02em;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 0;
            border-bottom: 1px solid var(--tf-border);
            padding-bottom: 0;
        }

        .stTabs [data-baseweb="tab"] {
            height: 42px;
            border-radius: 0;
            background: transparent;
            border: none;
            color: #64748b;
            padding: 0 0.9rem;
            margin-right: 0.2rem;
        }

        .stTabs [aria-selected="true"] {
            background: transparent;
            color: var(--tf-accent) !important;
            border-bottom: 2px solid var(--tf-accent);
        }

        .stButton button, .stDownloadButton button {
            border-radius: 13px;
            border: 1px solid var(--tf-border-strong);
            background: #ffffff;
            color: var(--tf-text);
            font-weight: 600;
            box-shadow: none;
            transition: transform 0.12s ease, box-shadow 0.12s ease, border-color 0.12s ease;
        }

        .stButton button:hover, .stDownloadButton button:hover {
            transform: translateY(-1px);
            box-shadow: 0 8px 18px rgba(18, 50, 75, 0.12);
            border-color: #9eb7ca;
        }

        .stButton button[kind="primary"] {
            background: var(--tf-accent);
            color: white;
            border-color: var(--tf-accent);
        }

        .stSelectbox label, .stTextInput label, .stRadio label, .stTextArea label {
            font-weight: 600;
        }

        [data-testid="stExpander"] {
            border: 1px solid var(--tf-border);
            border-radius: 12px;
            background: var(--tf-surface);
        }

        div[data-baseweb="select"] > div {
            border-color: var(--tf-border-strong);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _find_demo_ticket(
    runner: TicketFlowRunner,
    tickets: list[TicketRecord],
    category: str,
    *,
    missing_order: bool | None = None,
    tier: str | None = None,
    refundable: bool | None = None,
) -> TicketRecord | None:
    for ticket in tickets:
        if ticket.expected_category != category:
            continue
        if missing_order is True and ticket.linked_order_id:
            continue
        if missing_order is False and not ticket.linked_order_id:
            continue
        if tier is not None and ticket.customer_tier != tier:
            continue
        if refundable is not None:
            order = runner.repository.get_order_status(ticket.linked_order_id)
            if order is None or order.eligible_for_refund != refundable:
                continue
        return ticket
    return None


def _risk_candidate_count(runner: TicketFlowRunner, tickets: list[TicketRecord]) -> int:
    count = 0
    for ticket in tickets[:120]:
        order = runner.repository.get_order_status(ticket.linked_order_id)
        if ticket.expected_category == "billing_refund":
            count += 1
        elif ticket.expected_category == "technical_issue" and ticket.customer_tier == "enterprise":
            count += 1
        elif ticket.expected_category == "delivery_issue" and order is not None:
            if order.status == "pending" or (order.delivered_days_ago or 0) > 5:
                count += 1
    return count


def _service_badge(status: dict[str, Any]) -> str:
    if status.get("running"):
        return _badge("运行中", "success")
    return _badge("未启动", "warning")


def _render_minimind_control_panel(project_root: Path) -> None:
    st.sidebar.markdown("**本地模型控制台**")
    probe_enabled = bool(st.session_state.get("minimind_probe_requested"))
    try:
        report = runtime_report(project_root, probe=probe_enabled)
    except Exception as exc:  # noqa: BLE001 - presentation UI should fail softly.
        st.sidebar.error("本地模型状态读取失败")
        st.sidebar.caption(str(exc))
        return

    for item in report:
        probe = item.get("probe") or {}
        probe_note = "接口已探活" if probe.get("ok") else probe.get("message") if probe else "端口级检测"
        st.sidebar.markdown(
            f"""
            <div class="tf-service-row">
                <div class="tf-service-name">{item['label']} { _service_badge(item) }</div>
                <div class="tf-service-meta">
                    模型：{item['model']}<br>
                    接口：{item['base_url']}<br>
                    状态：{_zh_text(probe_note)}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    start_col, check_col, stop_col = st.sidebar.columns(3)
    if start_col.button("启动", use_container_width=True, help="启动结构化模型和回复模型"):
        try:
            results = start_all(project_root)
            st.session_state.minimind_last_action = results
            st.session_state.minimind_probe_requested = False
            st.toast("已发起 MiniMind 启动，模型加载需要几十秒。")
        except Exception as exc:  # noqa: BLE001
            st.session_state.minimind_last_action = [{"status": "failed", "message": str(exc)}]
            st.toast("MiniMind 启动失败，请查看日志。")
        st.rerun()
    if check_col.button("检测", use_container_width=True, help="对本地 OpenAI-compatible 接口做一次探活"):
        st.session_state.minimind_probe_requested = True
        st.rerun()
    if stop_col.button("停止", use_container_width=True, help="停止本控制台启动的进程，或安全匹配的 MiniMind 端口进程"):
        try:
            st.session_state.minimind_last_action = stop_all(project_root)
            st.session_state.minimind_probe_requested = False
            st.toast("已请求停止 MiniMind 本地模型服务。")
        except Exception as exc:  # noqa: BLE001
            st.session_state.minimind_last_action = [{"status": "failed", "message": str(exc)}]
            st.toast("停止 MiniMind 时遇到问题。")
        st.rerun()

    last_action = st.session_state.get("minimind_last_action")
    if last_action:
        with st.sidebar.expander("查看最近一次模型操作", expanded=False):
            st.json(last_action)
    st.sidebar.caption("说明：控制台只启动固定 LoRA、固定端口；不会执行任意命令。")


def _health_card(label: str, value: str, caption: str, tone: str = "neutral") -> str:
    return f"""
    <div class="tf-health-item">
        <div class="tf-health-label">{label}</div>
        <div class="tf-health-value">{_badge(value, tone)}</div>
        <div class="tf-health-caption">{caption}</div>
    </div>
    """


def _read_report(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _format_rate(value: Any, *, signed: bool = False) -> str:
    try:
        number = float(value)
    except Exception:
        return "待生成"
    prefix = "+" if signed and number >= 0 else ""
    return f"{prefix}{number * 100:.1f}%"


def _format_decimal(value: Any) -> str:
    try:
        return f"{float(value):.3f}"
    except Exception:
        return "待生成"


def _benchmark_report_cards(project_root: Path) -> list[dict[str, Any]]:
    reports_dir = project_root / "reports"
    retrieval = (
        _read_report(reports_dir / "retrieval_strategy_eval_a40_bge_source_lane_v2_full600.json")
        or _read_report(reports_dir / "retrieval_strategy_eval_a40_bge_source_lane_v1.json")
        or _read_report(reports_dir / "retrieval_strategy_eval_a40_bge_v8_stat1_full600.json")
        or _read_report(reports_dir / "retrieval_strategy_eval_a40_bge_v8.json")
        or _read_report(reports_dir / "retrieval_strategy_eval_a40_bge_v7.json")
        or _read_report(reports_dir / "retrieval_strategy_eval_a40_bge_v6.json")
    )
    multimodal = (
        _read_report(reports_dir / "multimodal_ablation_a40_bge_v2_stat1_full200.json")
        or _read_report(reports_dir / "multimodal_ablation_a40_bge_v2.json")
        or _read_report(reports_dir / "multimodal_ablation_a40_bge_v1.json")
    )

    retrieval_metrics = (retrieval or {}).get("metrics", {})
    retrieval_stack = (retrieval or {}).get("retrieval_stack", {})
    retrieval_caption = (
        f"{retrieval_stack.get('embedding_model', 'BGE-M3')} + "
        f"{str(retrieval_stack.get('fusion_method', 'RRF')).upper()} + "
        f"{retrieval_stack.get('reranker_backend', 'BGE-Reranker')}"
    )
    multimodal_lift = (multimodal or {}).get("lift", {})
    enabled_metrics = ((multimodal or {}).get("enabled", {}) or {}).get("metrics", {})

    return [
        {
            "title": "检索增强专项评测",
            "status": "A40 已完成" if retrieval else "待生成",
            "tone": "success" if retrieval else "warning",
            "caption": retrieval_caption if retrieval else "等待 A40 检索增强评测报告。",
            "metrics": {
                "HitRate": _format_rate(retrieval_metrics.get("hit_rate")),
                "MRR": _format_decimal(retrieval_metrics.get("mrr")),
                "错误率": _format_rate(retrieval_metrics.get("error_rate")),
            },
        },
        {
            "title": "多模态附件消融评测",
            "status": "A40 已完成" if multimodal else "待生成",
            "tone": "success" if multimodal else "warning",
            "caption": "同一批工单对比“启用附件证据”和“禁用附件证据”，衡量附件对证据链的真实增益。",
            "metrics": {
                "证据召回增益": _format_rate(multimodal_lift.get("multimodal_evidence_recall"), signed=True),
                "实体抽取增益": _format_rate(multimodal_lift.get("attachment_entity_accuracy"), signed=True),
                "启用后动作准确率": _format_rate(enabled_metrics.get("evidence_based_action_accuracy")),
            },
        },
    ]


def _render_benchmark_overview(project_root: Path) -> None:
    cards = _benchmark_report_cards(project_root)
    st.markdown('<div class="tf-section-label">专项评测看板</div>', unsafe_allow_html=True)
    columns = st.columns(len(cards))
    for column, card in zip(columns, cards, strict=True):
        metrics_html = "".join(
            f'<div class="tf-key">{name}</div><div class="tf-value">{value}</div>'
            for name, value in card["metrics"].items()
        )
        column.markdown(
            f"""
            <div class="tf-panel">
                <div class="tf-panel-title">{card["title"]}</div>
                <div>{_badge(card["status"], card["tone"])}</div>
                <div class="tf-caption" style="margin-top:0.55rem;">{card["caption"]}</div>
                <div class="tf-key-value" style="margin-top:0.75rem;">{metrics_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _render_system_overview(runner: TicketFlowRunner) -> None:
    settings = runner.settings
    project_root = _project_root()
    local_services = runtime_report(project_root, probe=False)
    local_running = sum(1 for item in local_services if item.get("running"))
    local_value = f"{local_running}/2 运行"
    local_tone = "success" if local_running == 2 else "warning" if local_running else "danger"
    email_ready = bool((settings.incident_email_to or settings.kb_ops_email_to) and not _missing_email_settings(settings))
    email_value = "已接入" if email_ready else "待配置"
    email_tone = "success" if email_ready else "warning"
    rag_value = "已开启" if settings.rag_enabled else "未开启"
    rag_tone = "success" if settings.rag_enabled else "danger"
    if settings.model_backend == "cloud_api":
        model_value = settings.cloud_model_name or DEFAULT_DEEPSEEK_MODEL
        model_caption = f"DeepSeek 云端接口：{settings.cloud_api_base_url or DEFAULT_DEEPSEEK_BASE_URL}"
        model_tone = "success" if settings.cloud_llm_enabled else "warning"
    else:
        model_value = local_value
        model_caption = "结构化服务与回复服务分离，支撑本地双 LoRA 演示。"
        model_tone = local_tone

    cards = [
        _health_card("模型后端", model_value, model_caption, model_tone),
        _health_card("证据检索", rag_value, "知识库、策略条款和历史工单进入统一证据链。", rag_tone),
        _health_card("外部邮箱", email_value, f"升级通知与知识候选发送至 {_mask_email(settings.incident_email_to or settings.kb_ops_email_to)}。", email_tone),
        _health_card("治理链路", "已启用", "证据充分性、工具级审批、回复事实校验全链路留痕。", "success"),
    ]
    for column, card in zip(st.columns(4), cards, strict=True):
        column.markdown(card, unsafe_allow_html=True)


def _filtered_tickets(
    tickets: list[TicketRecord],
    category_filter: str,
    tier_filter: str,
    channel_filter: str,
    keyword: str,
) -> list[TicketRecord]:
    keyword = keyword.strip().lower()
    filtered: list[TicketRecord] = []
    for ticket in tickets:
        if category_filter != "全部" and _label(CATEGORY_LABELS, ticket.expected_category) != category_filter:
            continue
        if tier_filter != "全部" and _label(TIER_LABELS, ticket.customer_tier) != tier_filter:
            continue
        if channel_filter != "全部" and _label(CHANNEL_LABELS, ticket.channel) != channel_filter:
            continue
        haystack = f"{ticket.ticket_id} {ticket.title} {ticket.body} {ticket.product}".lower()
        if keyword and keyword not in haystack:
            continue
        filtered.append(ticket)
    return filtered


def _ticket_option_label(ticket: TicketRecord) -> str:
    return (
        f"{ticket.ticket_id}｜{_label(CATEGORY_LABELS, ticket.expected_category)}｜"
        f"{_label(TIER_LABELS, ticket.customer_tier)}｜{ticket.title}"
    )


def _panel(title: str, body: str = "", badges: list[str] | None = None) -> None:
    badges_html = "".join(badges or [])
    title = _zh_text(title)
    body = _zh_text(body)
    html = f'<div class="tf-panel"><div class="tf-panel-title">{title}</div>'
    if badges_html:
        html += f"<div>{badges_html}</div>"
    if body:
        html += f'<div class="tf-caption">{body}</div>'
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def _render_ticket_summary(ticket: TicketRecord) -> None:
    st.markdown(
        f"""
        <div class="tf-panel">
            <div class="tf-panel-title">工单概览</div>
            <div>
                {_badge(_label(CATEGORY_LABELS, ticket.expected_category), "accent")}
                {_badge(_label(TIER_LABELS, ticket.customer_tier), "neutral")}
                {_badge(_label(CHANNEL_LABELS, ticket.channel), "success")}
            </div>
            <div class="tf-caption" style="margin-top:0.65rem;">{ticket.title}</div>
            <div class="tf-key-value">
                <div class="tf-key">工单编号</div><div class="tf-value">{ticket.ticket_id}</div>
                <div class="tf-key">产品</div><div class="tf-value">{ticket.product}</div>
                <div class="tf-key">当前状态</div><div class="tf-value">{_label(STATUS_LABELS, ticket.status)}</div>
                <div class="tf-key">创建时间</div><div class="tf-value">{ticket.created_at.strftime("%Y-%m-%d %H:%M")}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="tf-section-label">客户描述</div>', unsafe_allow_html=True)
    with st.container(border=True):
        st.write(_zh_text(ticket.body))
        if ticket.source_dataset:
            with st.expander("查看样本来源", expanded=False):
                st.caption("当前演示工单文本来自公开 support ticket 语料的中文化改写版本。")
                st.write(
                    {
                        "来源数据集": ticket.source_dataset,
                        "来源语种": ticket.source_language or "未知",
                        "来源队列": ticket.source_queue or "未知",
                        "来源主题": ticket.source_subject or "未知",
                    }
                )


def _render_trace(state: dict[str, Any]) -> None:
    trace_items = state.get("trace", [])
    if not trace_items:
        st.info("工单进入处理流程后，这里会显示各节点的处理记录。")
        return
    for idx, item in enumerate(trace_items, start=1):
        actor = _label(ACTOR_LABELS, item["actor"])
        step = _label(STEP_LABELS, item["step"])
        with st.container(border=True):
            st.markdown(f"**{idx}. {step}**")
            st.caption(actor)
            st.write(_zh_text(item["detail"]))
            if item.get("payload"):
                with st.expander("查看节点原始数据", expanded=False):
                    st.json(item["payload"])


def _render_tool_calls(state: dict[str, Any]) -> None:
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        st.info("当前工单尚未产生工具调用。")
        return
    for idx, tool_call in enumerate(tool_calls, start=1):
        with st.container(border=True):
            st.markdown(f"**{idx}. {_label(TOOL_LABELS, tool_call['tool'])}**")
            if tool_call.get("summary"):
                st.write(_zh_text(tool_call["summary"]))
            if tool_call.get("args"):
                st.caption("请求参数")
                st.json(tool_call["args"])
            if tool_call.get("result") is not None:
                with st.expander("查看返回结果", expanded=False):
                    st.json(tool_call["result"])
            elif tool_call:
                with st.expander("查看原始记录", expanded=False):
                    st.json(tool_call)


def _resolve_attachment_storage_path(storage_path: str | None) -> Path | None:
    if not storage_path:
        return None
    raw_path = Path(storage_path)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        project_root = Path(__file__).resolve().parents[2]
        candidates.extend([Path.cwd() / raw_path, project_root / raw_path])
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _render_attachment_evidence(state: dict[str, Any]) -> None:
    attachments = state.get("attachments", [])
    evidence = state.get("attachment_evidence", [])
    if not attachments and not evidence:
        st.info("当前工单没有上传附件。多模态工单会在这里展示原始附件、OCR 文本、视觉摘要和结构化字段。")
        return

    if attachments:
        st.markdown("**原始附件**")
        for attachment in attachments:
            with st.container(border=True):
                st.markdown(f"**{attachment.filename}**")
                st.caption(
                    f"类型：{attachment.file_type} ｜ 来源：{attachment.source_dataset or '工单上传'} ｜ 解析状态：{attachment.parse_status}"
                )
                storage_path = _resolve_attachment_storage_path(attachment.storage_path)
                if storage_path and attachment.file_type == "image":
                    st.image(str(storage_path), caption=f"原始附件预览：{attachment.filename}", use_container_width=True)
                    st.caption(f"文件路径：{storage_path}")
                elif storage_path:
                    st.caption(f"原始文件路径：{storage_path}")
                else:
                    st.caption("当前记录没有可预览的原始文件，仅展示已保存的解析结果。")
                if attachment.visual_summary:
                    st.write(attachment.visual_summary)
                if attachment.ocr_text:
                    with st.expander("查看 OCR 文本", expanded=False):
                        st.write(attachment.ocr_text)

    if evidence:
        st.markdown("**附件证据**")
        for item in evidence:
            with st.container(border=True):
                badges = [
                    _badge(f"证据类型：{_label(ATTACHMENT_EVIDENCE_TYPE_LABELS, item.evidence_type)}", "accent"),
                    _badge(f"置信度：{item.confidence}", "success" if item.confidence >= 0.7 else "warning"),
                ]
                if item.risk_flags:
                    badges.append(_badge("需要人工复核", "warning"))
                st.markdown(
                    f"""
                    <div class="tf-panel-title">{item.evidence_id}</div>
                    <div>{''.join(badges)}</div>
                    <div class="tf-caption" style="margin-top:0.6rem;">{item.visual_summary or item.extracted_text or '未抽取到可靠文本。'}</div>
                    """,
                    unsafe_allow_html=True,
                )
                if item.entities:
                    st.caption("结构化字段")
                    st.json(item.entities)


def _render_docs(state: dict[str, Any]) -> None:
    docs = state.get("retrieved_docs", [])
    if not docs:
        st.info("处理过程中命中的知识库、策略和历史案例会显示在这里。")
        return
    retrieval_stats = state.get("retrieval_stats")
    if retrieval_stats:
        final_hits = retrieval_stats.final_hits if hasattr(retrieval_stats, "final_hits") else retrieval_stats.get("final_hits", 0)
        lexical_hits = retrieval_stats.lexical_hits if hasattr(retrieval_stats, "lexical_hits") else retrieval_stats.get("lexical_hits", 0)
        vector_hits = retrieval_stats.vector_hits if hasattr(retrieval_stats, "vector_hits") else retrieval_stats.get("vector_hits", 0)
        used_vector = retrieval_stats.used_vector if hasattr(retrieval_stats, "used_vector") else retrieval_stats.get("used_vector", False)
        fallback_to_keywords = retrieval_stats.fallback_to_keywords if hasattr(retrieval_stats, "fallback_to_keywords") else retrieval_stats.get("fallback_to_keywords", False)
        source_counts = retrieval_stats.source_counts if hasattr(retrieval_stats, "source_counts") else retrieval_stats.get("source_counts", {})
        stats_badges = [
            _badge(f"最终证据：{final_hits}", "accent"),
            _badge(f"关键词召回：{lexical_hits}", "neutral"),
            _badge(f"向量召回：{vector_hits}", "success" if used_vector else "neutral"),
        ]
        for source_name in ("kb", "policy", "history"):
            if source_counts.get(source_name):
                stats_badges.append(_badge(f"{_label(SOURCE_LABELS, source_name)}：{source_counts[source_name]}", "neutral"))
        if fallback_to_keywords:
            stats_badges.append(_badge("向量层回退到关键词", "warning"))
        _panel("检索结果", "", stats_badges)
    for doc in docs:
        badges = [
            _badge(_label(SOURCE_LABELS, doc.source_type), "accent"),
            _badge(f"相关分：{doc.score}", "neutral"),
        ]
        retrieval_mode = doc.metadata.get("retrieval")
        if retrieval_mode:
            badges.append(_badge(f"召回方式：{_label(RETRIEVAL_MODE_LABELS, retrieval_mode)}", "neutral"))
        st.markdown(
            f"""
            <div class="tf-panel">
                <div class="tf-panel-title">{doc.title}</div>
                <div>{''.join(badges)}</div>
                <div class="tf-caption" style="margin-top:0.6rem;">{doc.snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if doc.citations:
            st.caption("引用：" + "、".join(_doc_id_label(citation.source_path) for citation in doc.citations))


def _render_audit_log(state: dict[str, Any]) -> None:
    audit_log = state.get("audit_log", [])
    if not audit_log:
        st.info("关键决策和执行动作会记录在这里。")
        return
    for event in audit_log:
        with st.container(border=True):
            st.markdown(f"**{_label(ACTOR_LABELS, event.actor)}** · `{event.event_type}`")
            st.caption(str(event.timestamp))
            st.write(_zh_text(event.detail))
            if event.payload:
                with st.expander("查看审计负载", expanded=False):
                    st.json(event.payload)


def _render_external_ops(state: dict[str, Any]) -> None:
    records = state.get("external_ops", [])
    if not records:
        st.info("外部通知和知识沉淀结果会显示在这里。")
        return
    op_labels = {
        "incident_email": "升级通知邮件",
        "kb_candidate_email": "知识候选邮件",
    }
    for record in records:
        status = record.status if hasattr(record, "status") else record.get("status", "unknown")
        op_type = record.op_type if hasattr(record, "op_type") else record.get("op_type", "unknown")
        recipient = record.recipient if hasattr(record, "recipient") else record.get("recipient", "-")
        subject = record.subject if hasattr(record, "subject") else record.get("subject", "")
        error_message = record.error_message if hasattr(record, "error_message") else record.get("error_message")
        tone = "success" if status == "sent" else "warning" if status == "skipped" else "danger"
        _panel(
            op_labels.get(op_type, op_type),
            error_message or subject,
            [
                _badge(op_labels.get(op_type, op_type), "accent"),
                _badge(_label(EXECUTION_STATUS_LABELS, status), tone),
                _badge(recipient, "neutral"),
            ],
        )
        with st.expander("查看发送详情", expanded=False):
            st.json(record.model_dump(mode="json") if hasattr(record, "model_dump") else record)


def _render_triage_card(state: dict[str, Any]) -> None:
    triage = state.get("triage_result")
    if not triage:
        _panel("分诊结果", "工作流运行后，会在这里展示类别、优先级和服务时效风险。")
        return
    badges = [
        _badge(_label(CATEGORY_LABELS, triage.category), "accent"),
        _badge(f"优先级：{_label(PRIORITY_LABELS, triage.priority)}", "warning" if triage.priority in {"high", "urgent"} else "neutral"),
        _badge("服务时效风险" if triage.sla_risk else "服务时效正常", "danger" if triage.sla_risk else "success"),
        _source_badge(getattr(triage, "decision_source", "rule")),
    ]
    _panel("分诊结果", _triage_summary(triage, state.get("ticket")), badges)
    st.caption(f"置信度：{triage.confidence}")
    if getattr(triage, "fallback_reason", None):
        st.caption(f"回退说明：{_zh_text(triage.fallback_reason)}")


def _render_action_card(state: dict[str, Any]) -> None:
    action = state.get("proposed_action")
    if not action:
        _panel("动作建议", "系统会在拉取证据后生成下一步动作建议。")
        return
    badges = [
        _badge(_label(ACTION_LABELS, action.action_type), "accent"),
        _badge("需要审批" if action.requires_approval else "无需审批", "danger" if action.requires_approval else "success"),
        _badge(_label(TOOL_LABELS, action.suggested_tool), "neutral"),
        _source_badge(getattr(action, "decision_source", "rule")),
    ]
    _panel("动作建议", _action_summary(action, state.get("sufficiency_result")), badges)
    if getattr(action, "fallback_reason", None):
        st.caption(f"回退说明：{_zh_text(action.fallback_reason)}")
    with st.expander("查看动作详情", expanded=False):
        st.json(action.model_dump(mode="json"))


def _render_execution_card(state: dict[str, Any]) -> None:
    execution_result = state.get("execution_result")
    if not execution_result:
        _panel("执行结果", "审批通过后，这里会展示执行结果。")
        return
    tone = "success"
    if execution_result["status"] in {"needs_handoff", "rejected"}:
        tone = "danger"
    _panel(
        "执行结果",
        execution_result.get("error") or "动作已执行完成。",
        [
            _badge(f"状态：{_label(EXECUTION_STATUS_LABELS, execution_result['status'])}", tone),
            _badge(_label(TOOL_LABELS, execution_result.get("tool_name") or "无工具执行"), "neutral"),
        ],
    )
    with st.expander("查看执行详情", expanded=False):
        st.json(execution_result)


def _render_final_response(state: dict[str, Any]) -> None:
    if "draft_reply" not in state:
        _panel("客户回复", "执行完成后，会在这里展示最终客户回复和内部备注。")
        return
    reply = state["draft_reply"]
    st.markdown(
        f"""
        <div class="tf-panel">
            <div class="tf-panel-title">客户回复</div>
            <div>
                {_badge(_label(STATUS_LABELS, reply.status), "success")}
                {_source_badge(getattr(reply, "draft_source", "rule"))}
            </div>
            <div class="tf-caption" style="margin-top:0.75rem;">{reply.customer_reply}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if getattr(reply, "fallback_reason", None):
        st.caption(f"回退说明：{_zh_text(reply.fallback_reason)}")
    st.markdown("**内部备注**")
    with st.container(border=True):
        st.write(_reply_internal_summary(reply, state))
    if reply.citations:
        st.markdown("**引用证据**")
        for citation in reply.citations:
            st.caption(_doc_id_label(citation.source_path))


def _source_badge(source: str) -> str:
    mapping = {
        "rule": ("安全兜底", "neutral"),
        "llm_cloud": ("云端模型", "accent"),
        "llm_minimind": ("本地模型", "success"),
        "human_edit": ("人工修订", "warning"),
        "governance": ("治理路线", "warning"),
    }
    label, tone = mapping.get(source, (source, "neutral"))
    return _badge(label, tone)


def _render_sufficiency_card(state: dict[str, Any]) -> None:
    result = state.get("sufficiency_result")
    if not result:
        _panel("上下文证据充分性", "RAG 检索结束后，这里会展示当前上下文是否足够支撑高风险动作。")
        return
    tone = "success" if result.sufficient else "warning"
    badges = [
        _badge(f"动作路线：{_label(ROUTE_FAMILY_LABELS, result.route_family)}", "accent"),
        _badge("证据充分" if result.sufficient else "证据不足", tone),
        _source_badge(getattr(result, "decision_source", "rule")),
    ]
    _panel("上下文证据充分性", _sufficiency_summary(result), badges)
    if result.required_sources:
        st.caption("必需证据：" + _source_list_label(result.required_sources))
    if result.supporting_doc_ids:
        st.caption("已命中证据：" + "、".join(_doc_id_label(doc_id) for doc_id in result.supporting_doc_ids))
    if result.missing_sources:
        st.caption("缺失证据：" + _source_list_label(result.missing_sources))
    if getattr(result, "fallback_reason", None):
        st.caption(f"回退说明：{_zh_text(result.fallback_reason)}")


def _render_tool_approval_card(state: dict[str, Any], interrupt_payload: dict[str, Any] | None = None) -> None:
    tool_policy = state.get("tool_approval_policy")
    if not tool_policy and interrupt_payload:
        tool_policy = interrupt_payload.get("tool_policy")
    if not tool_policy:
        _panel("工具级审批", "敏感工具的审批策略和留痕会在这里显示。")
        return

    approval_mode = tool_policy.get("approval_mode") if isinstance(tool_policy, dict) else tool_policy.approval_mode
    sensitive = tool_policy.get("sensitive") if isinstance(tool_policy, dict) else tool_policy.sensitive
    description = tool_policy.get("description") if isinstance(tool_policy, dict) else tool_policy.description
    tool_name = tool_policy.get("tool_name") if isinstance(tool_policy, dict) else tool_policy.tool_name
    badges = [
        _badge(_label(TOOL_LABELS, tool_name), "accent"),
        _badge(_label(APPROVAL_MODE_LABELS, approval_mode), "warning" if sensitive else "neutral"),
        _badge("敏感工具" if sensitive else "自动执行", "danger" if sensitive else "success"),
    ]
    _panel("工具级审批", _zh_text(description or "Current tool approval policy."), badges)


def _render_fact_check_card(state: dict[str, Any]) -> None:
    fact_check = state.get("reply_fact_check")
    if not fact_check:
        _panel("回复事实校验", "回复生成后，这里会展示事实校验结果。")
        return
    badges = [
        _badge("校验通过" if fact_check.passed else "需要兜底", "success" if fact_check.passed else "danger"),
        _badge(f"证据支持比例：{fact_check.supported_claim_ratio}", "neutral"),
        _badge("已重写" if fact_check.rewritten else "未重写", "warning" if fact_check.rewritten else "neutral"),
        _badge("已降级" if fact_check.downgraded else "保留回复", "warning" if fact_check.downgraded else "success"),
    ]
    _panel("回复事实校验", _zh_text(fact_check.review_note), badges)
    if fact_check.unsupported_claims:
        st.caption("未被证据支持的断言：" + "、".join(_zh_text(claim) for claim in fact_check.unsupported_claims))


def _render_dashboard(runner: TicketFlowRunner, tickets: list[TicketRecord]) -> None:
    enterprise_count = sum(1 for ticket in tickets if ticket.customer_tier == "enterprise")
    refund_count = sum(1 for ticket in tickets if ticket.expected_category == "billing_refund")
    risk_count = _risk_candidate_count(runner, tickets)

    st.markdown(
        """
        <div class="tf-page-header">
            <div class="tf-hero-kicker">企业工单智能运维中心</div>
            <div class="tf-page-title">工单协同智能体工作台</div>
            <div class="tf-page-subtitle">
                面向客服与信息技术服务团队的企业运维台，统一呈现工单队列、证据检索、工具审批、本地模型运行状态与外部协同结果。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    _render_system_overview(runner)
    _render_benchmark_overview(_project_root())

    metric_cols = st.columns(4)
    metric_cols[0].metric("开放工单", len(tickets))
    metric_cols[1].metric("企业客户工单", enterprise_count)
    metric_cols[2].metric("风险关注工单", risk_count)
    metric_cols[3].metric("退款相关工单", refund_count)


def _render_model_backend_switch(runner: TicketFlowRunner) -> None:
    settings = runner.settings
    options = ["minimind_split", "cloud_api"]
    labels = {
        "minimind_split": "本地 MiniMind 双 LoRA",
        "cloud_api": "DeepSeek 云端模型",
    }
    current = st.session_state.get("ticketflow_selected_backend") or settings.model_backend
    if current not in options:
        current = "cloud_api"
    st.sidebar.markdown("**模型后端切换**")
    selected = st.sidebar.radio(
        "选择当前推理后端",
        options,
        index=options.index(current),
        format_func=lambda value: labels[value],
        horizontal=False,
        label_visibility="collapsed",
    )
    if selected != st.session_state.get("ticketflow_selected_backend", settings.model_backend):
        st.session_state.ticketflow_selected_backend = selected
        st.session_state.pop("ticketflow_runner_cache_key", None)
        st.rerun()

    if selected == "cloud_api":
        model_name = settings.cloud_model_name or DEFAULT_DEEPSEEK_MODEL
        base_url = settings.cloud_api_base_url or DEFAULT_DEEPSEEK_BASE_URL
        st.sidebar.caption(f"当前云端模型：DeepSeek / {model_name}")
        st.sidebar.caption(f"接口地址：{base_url}")
        if not settings.cloud_api_key:
            st.sidebar.warning("DeepSeek API Key 未配置，云端模式暂不可用。")
        elif settings.cloud_llm_enabled:
            st.sidebar.success("DeepSeek 云端模型已接入")
    else:
        st.sidebar.caption(f"结构化模型：{settings.minimind_structured_model or '未配置'}")
        st.sidebar.caption(f"回复模型：{settings.minimind_reply_model or '未配置'}")


def _run_ticket_for_ui(runner: Any, ticket_id: str) -> tuple[Any | None, str | None]:
    try:
        return runner.run_ticket(ticket_id), None
    except Exception as exc:
        exc_name = type(exc).__name__
        exc_text = str(exc)
        if "ConnectionError" in exc_name or "ConnectionError" in exc_text or "Timeout" in exc_name or "Timeout" in exc_text:
            reason = "模型服务连接异常"
            suggestion = "请检查当前模型后端是否可用，或切换到本地 MiniMind 双 LoRA 后重试。"
        else:
            reason = "工作流执行异常"
            suggestion = "请保留当前工单和运行环境，稍后查看后台日志定位具体节点。"
        return None, f"工作流运行失败：{reason}。{suggestion}（技术原因：{exc_name}）"


def _render_production_control_panel(runner: TicketFlowRunner) -> None:
    st.sidebar.markdown("**生产控制面**")
    try:
        tasks = runner.repository.list_workflow_tasks(limit=20)
        approvals = runner.repository.list_approval_requests(limit=20)
        outbox_events = runner.repository.list_outbox_events(limit=20)
    except Exception as exc:  # noqa: BLE001 - ops sidebar must fail softly.
        st.sidebar.warning("生产控制面状态读取失败")
        st.sidebar.caption(str(exc))
        return

    pending_tasks = sum(1 for item in tasks if item.get("status") in {"queued", "running"})
    pending_approvals = sum(1 for item in approvals if item.get("status") == "pending")
    pending_outbox = sum(1 for item in outbox_events if item.get("status") == "pending")
    st.sidebar.caption(f"任务队列：{pending_tasks} 个待处理 / 最近 {len(tasks)} 条")
    st.sidebar.caption(f"审批队列：{pending_approvals} 个待处理 / 最近 {len(approvals)} 条")
    st.sidebar.caption(f"Outbox：{pending_outbox} 个待投递 / 最近 {len(outbox_events)} 条")
    with st.sidebar.expander("查看生产控制面快照", expanded=False):
        st.write("最近任务")
        st.json(tasks[:5])
        st.write("最近审批")
        st.json(approvals[:5])
        st.write("最近 Outbox")
        st.json(outbox_events[:5])


def _render_production_control_panel(runner: TicketFlowRunner) -> None:
    st.sidebar.markdown("**生产控制面**")
    try:
        snapshot = _load_control_plane_snapshot(_project_root(), runner)
        tasks = snapshot["tasks"]
        approvals = snapshot["approvals"]
        outbox_events = snapshot["outbox_events"]
    except Exception as exc:  # noqa: BLE001 - ops sidebar must fail softly.
        st.sidebar.warning("生产控制面状态读取失败。")
        st.sidebar.caption(str(exc))
        return

    if snapshot["source"] == "api":
        st.sidebar.success("API 服务：已连接")
    else:
        st.sidebar.warning("API 服务：未连接，已回退本地仓储")
        st.sidebar.caption(f"目标地址：{snapshot['api_base_url']}")
        if snapshot.get("error"):
            st.sidebar.caption(f"回退原因：{snapshot['error']}")

    ready = snapshot.get("ready") or {}
    if isinstance(ready, dict) and ready:
        database = ready.get("database") if isinstance(ready.get("database"), dict) else {}
        celery = ready.get("dependencies", {}).get("celery", {}) if isinstance(ready.get("dependencies"), dict) else {}
        st.sidebar.caption(f"API 地址：{snapshot['api_base_url']}")
        st.sidebar.caption(f"数据后端：{database.get('backend', 'unknown')}")
        st.sidebar.caption(f"Celery：{'队列模式' if not celery.get('task_always_eager') else '本地同步模式'}")

    pending_tasks = sum(1 for item in tasks if item.get("status") in {"queued", "running", "waiting_approval"})
    pending_approvals = sum(1 for item in approvals if item.get("status") == "pending")
    pending_outbox = sum(1 for item in outbox_events if item.get("status") == "pending")
    st.sidebar.caption(f"任务队列：{pending_tasks} 个待处理 / 最近 {len(tasks)} 条")
    st.sidebar.caption(f"审批队列：{pending_approvals} 个待处理 / 最近 {len(approvals)} 条")
    st.sidebar.caption(f"Outbox：{pending_outbox} 个待投递 / 最近 {len(outbox_events)} 条")
    with st.sidebar.expander("查看生产控制面快照", expanded=False):
        st.write("最近任务")
        st.json(tasks[:5])
        st.write("最近审批")
        st.json(approvals[:5])
        st.write("最近 Outbox")
        st.json(outbox_events[:5])


def _render_runtime_mode(runner: TicketFlowRunner) -> None:
    settings = runner.settings
    _render_model_backend_switch(runner)
    st.sidebar.markdown("**运行环境**")
    st.sidebar.caption(f"模型后端：{_label(MODEL_BACKEND_LABELS, settings.model_backend)}")
    st.sidebar.caption(f"知识检索：{'开启' if settings.rag_enabled else '关闭'}")
    st.sidebar.caption(f"升级通知邮箱：{_mask_email(settings.incident_email_to)}")
    st.sidebar.caption(f"知识运营邮箱：{_mask_email(settings.kb_ops_email_to)}")
    missing_email_settings = _missing_email_settings(settings)
    if settings.incident_email_to or settings.kb_ops_email_to:
        if missing_email_settings:
            st.sidebar.warning("外部邮箱：收件人已接入，发信授权待完成")
            with st.sidebar.expander("查看邮箱接入状态", expanded=False):
                st.write(f"升级通知收件人：{settings.incident_email_to or '未配置'}")
                st.write(f"知识运营收件人：{settings.kb_ops_email_to or '未配置'}")
                st.write("待补充：" + "、".join(missing_email_settings))
                st.caption("QQ 邮箱真实发信需要在邮箱设置中开启发信服务，并使用授权码作为发信凭据。")
        else:
            st.sidebar.success("外部邮箱：真实发信已就绪")
    if settings.model_backend == "minimind_split" and settings.minimind_split_enabled:
        st.sidebar.success("当前：本地双适配模型模式")
        with st.sidebar.expander("查看本地模型接口", expanded=False):
            st.caption(f"结构化模型标识：{settings.minimind_structured_model}")
            st.caption(f"结构化接口：{settings.minimind_structured_base_url}")
            st.caption(f"回复模型标识：{settings.minimind_reply_model}")
            st.caption(f"回复接口：{settings.minimind_reply_base_url}")
    elif settings.model_backend == "minimind_api" and settings.minimind_enabled:
        st.sidebar.success("当前：本地模型优先")
        st.sidebar.caption(f"模型：{settings.minimind_model}")
        st.sidebar.caption(f"接口：{settings.minimind_base_url}")
    elif settings.llm_enabled:
        st.sidebar.success("当前：云端模型模式")
        st.sidebar.caption(f"模型：{settings.cloud_model_name}")
        st.sidebar.caption(f"接口：{settings.cloud_api_base_url}{settings.cloud_api_chat_path}")
    elif settings.llm_provider_preset:
        st.sidebar.warning("当前：云端模型待激活")
        st.sidebar.caption(f"模型：{settings.cloud_model_name}")
        st.sidebar.caption("检测到模型预设，但缺少可用凭据；正式模式需要先连接大模型。")
    else:
        st.sidebar.error("当前：未连接模型")
        st.sidebar.caption("正式模式仅支持大模型链路；离线兜底仅用于开发自检，不建议作为演示配置。")


def main() -> None:
    st.set_page_config(
        page_title="TicketFlow 工单处理平台",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_styles()

    runner = _runner()
    all_tickets = runner.list_open_tickets(limit=500)

    st.sidebar.title("工作台设置")
    _render_runtime_mode(runner)
    _render_production_control_panel(runner)
    _render_minimind_control_panel(_project_root())
    if st.sidebar.button("重置演示数据", type="primary"):
        runner.reset_demo_data()
        st.session_state.pop("last_result", None)
        st.session_state.pop("selected_ticket_id", None)
        st.rerun()

    _render_dashboard(runner, all_tickets)

    with st.sidebar.expander("快捷定位工单", expanded=False):
        for label, config in SCENARIO_CONFIG.items():
            ticket = _find_demo_ticket(runner, all_tickets, **config)
            disabled = ticket is None
            help_text = ticket.title if ticket is not None else "当前队列中没有匹配工单，请先重置数据。"
            if st.button(label, disabled=disabled, help=help_text, use_container_width=True):
                st.session_state.selected_ticket_id = ticket.ticket_id
                st.session_state.pop("last_result", None)
                st.rerun()

    st.markdown("#### 工单队列")
    filter_cols = st.columns([1.05, 0.95, 0.95, 1.25])
    category_filter = filter_cols[0].selectbox(
        "工单类别",
        ["全部"] + sorted({_label(CATEGORY_LABELS, ticket.expected_category) for ticket in all_tickets}),
    )
    tier_filter = filter_cols[1].selectbox(
        "客户等级",
        ["全部"] + sorted({_label(TIER_LABELS, ticket.customer_tier) for ticket in all_tickets}),
    )
    channel_filter = filter_cols[2].selectbox(
        "来源渠道",
        ["全部"] + sorted({_label(CHANNEL_LABELS, ticket.channel) for ticket in all_tickets}),
    )
    keyword = filter_cols[3].text_input("关键词搜索", placeholder="工单标题、产品、正文关键词")

    tickets = _filtered_tickets(all_tickets, category_filter, tier_filter, channel_filter, keyword)
    ticket_map = {ticket.ticket_id: ticket for ticket in tickets}
    active_result_ticket_id = st.session_state.get("active_ticket_for_result")
    if active_result_ticket_id and active_result_ticket_id not in ticket_map and st.session_state.get("last_result") is not None:
        try:
            completed_ticket = runner.get_ticket(active_result_ticket_id)
        except Exception:
            completed_ticket = None
        if completed_ticket is not None:
            tickets = [completed_ticket] + tickets
            ticket_map[completed_ticket.ticket_id] = completed_ticket

    if not ticket_map:
        st.warning("当前筛选条件下没有工单。你可以清空筛选或点击左侧的“重置演示数据”。")
        return

    default_ticket_id = st.session_state.get("selected_ticket_id")
    if default_ticket_id not in ticket_map:
        default_ticket_id = tickets[0].ticket_id
        st.session_state.selected_ticket_id = default_ticket_id

    options = [ticket.ticket_id for ticket in tickets]
    selected_ticket_id = st.selectbox(
        "选择一个工单进入工作流",
        options,
        index=options.index(default_ticket_id),
        format_func=lambda ticket_id: _ticket_option_label(ticket_map[ticket_id]),
    )
    if st.session_state.get("active_ticket_for_result") and st.session_state["active_ticket_for_result"] != selected_ticket_id:
        st.session_state.pop("last_result", None)
        st.session_state.pop("workflow_error", None)
    st.session_state.selected_ticket_id = selected_ticket_id
    selected_ticket = ticket_map[selected_ticket_id]

    action_cols = st.columns([1.2, 1.0, 4.0])
    if action_cols[0].button("运行工作流", type="primary", use_container_width=True):
        run_result, workflow_error = _run_ticket_for_ui(runner, selected_ticket.ticket_id)
        st.session_state.workflow_error = workflow_error
        st.session_state.last_result = run_result
        if run_result is not None:
            st.session_state.active_ticket_for_result = selected_ticket.ticket_id
    if action_cols[1].button("刷新工单队列", use_container_width=True):
        st.rerun()
    action_cols[2].caption(f"当前选中：{selected_ticket.ticket_id}｜{selected_ticket.title}")
    if st.session_state.get("workflow_error"):
        st.error(st.session_state.workflow_error)

    left, middle, right = st.columns([1.05, 1.45, 1.2])
    result = st.session_state.get("last_result")
    state: dict[str, Any] = result.state if result is not None else {}

    with left:
        st.markdown("#### 工单详情")
        _render_ticket_summary(selected_ticket)

    with middle:
        st.markdown("#### 处理记录")
        if state:
            st.caption(f"线程 ID：`{state.get('thread_id', '-')}`")
        tabs = st.tabs(["流转记录", "工具日志", "附件证据", "关联知识", "外部协同", "审计记录"])
        with tabs[0]:
            _render_trace(state)
        with tabs[1]:
            _render_tool_calls(state)
        with tabs[2]:
            _render_attachment_evidence(state)
        with tabs[3]:
            _render_docs(state)
        with tabs[4]:
            _render_external_ops(state)
        with tabs[5]:
            _render_audit_log(state)

    with right:
        st.markdown("#### 处理结果")
        _render_triage_card(state)
        _render_sufficiency_card(state)
        _render_action_card(state)

        if result is not None and result.interrupted:
            payload = result.interrupt_payload or {}
            proposed = ActionProposal.model_validate(payload["proposed_action"])
            _render_tool_approval_card(state, payload)
            st.warning("该工单当前处于审批等待状态。")
            with st.expander("查看审批负载", expanded=False):
                st.json(payload)
            with st.form("approval_form"):
                decision = st.radio(
                    "审批决定",
                    list(payload.get("tool_policy", {}).get("allowed_decisions", [])) or ["approve", "edit", "reject"],
                    format_func=lambda value: _label(DECISION_LABELS, value),
                    horizontal=True,
                )
                comment = st.text_area("审批备注", value="已完成业务审核。")
                edited_action = proposed
                if decision == "edit":
                    action_type = st.selectbox(
                        "修改动作",
                        ["refund", "escalation", "request_info", "status_update", "troubleshoot"],
                        format_func=lambda value: _label(ACTION_LABELS, value),
                    )
                    edited_tool = (
                        "issue_refund_request"
                        if action_type == "refund"
                        else "create_escalation"
                        if action_type == "escalation"
                        else "update_ticket_status"
                    )
                    edited_tool_args = dict(proposed.tool_args)
                    if action_type == "request_info":
                        edited_tool_args = {"ticket_id": selected_ticket.ticket_id, "status": "waiting_on_customer"}
                    elif action_type == "status_update":
                        edited_tool_args = {"ticket_id": selected_ticket.ticket_id, "status": "in_progress"}
                    elif action_type == "troubleshoot":
                        edited_tool_args = {"ticket_id": selected_ticket.ticket_id, "status": "investigating"}
                    elif action_type == "escalation":
                        edited_tool_args = {
                            "ticket_id": selected_ticket.ticket_id,
                            "reason": "人工审批后转专家团队继续处理。",
                            "priority": state.get("triage_result").priority if state.get("triage_result") else "high",
                        }
                    edited_action = ActionProposal(
                        action_type=action_type,
                        rationale=f"人工审批将动作从 {proposed.action_type} 调整为 {action_type}。",
                        requires_approval=True,
                        suggested_tool=edited_tool,
                        tool_args=edited_tool_args,
                        confidence=proposed.confidence,
                        target_status=proposed.target_status,
                        route_family=proposed.route_family,
                        sufficiency_passed=proposed.sufficiency_passed,
                        supporting_doc_ids=proposed.supporting_doc_ids,
                        missing_sources=proposed.missing_sources,
                        decision_source="human_edit",
                    )
                submitted = st.form_submit_button("提交审批并继续")
            if submitted:
                decision_payload = {
                    "decision": decision,
                    "edited_action": edited_action.model_dump(mode="json") if decision == "edit" else None,
                    "comment": comment,
                }
                st.session_state.last_result = runner.resume_ticket(state["thread_id"], decision_payload)
                st.session_state.active_ticket_for_result = selected_ticket.ticket_id
                st.rerun()
        else:
            approval_state = state.get("approval_state")
            if approval_state is not None:
                _panel(
                    "审批状态",
                    "",
                    [_badge(_label(APPROVAL_STATE_LABELS, approval_state), "warning" if approval_state == "pending_review" else "success")],
                )
            else:
                _panel("审批状态", "工作流运行后显示当前审批阶段。")
            _render_tool_approval_card(state)

        _render_execution_card(state)
        _render_fact_check_card(state)
        _render_final_response(state)


if __name__ == "__main__":
    main()
