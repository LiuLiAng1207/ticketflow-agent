from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

try:
    from .graph import TicketFlowRunner
    from .models import ActionProposal, ReviewDecision, TicketRecord
except ImportError:
    from ticketflow.graph import TicketFlowRunner
    from ticketflow.models import ActionProposal, ReviewDecision, TicketRecord


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

SOURCE_LABELS = {
    "kb": "知识条目",
    "policy": "策略条款",
    "history": "历史案例",
    "customer": "客户画像",
    "order": "订单信息",
    "template": "回复模板",
}

ACTOR_LABELS = {
    "supervisor": "流程编排服务",
    "triage_agent": "分诊服务",
    "knowledge_agent": "知识检索服务",
    "resolution_agent": "处置决策服务",
    "manager": "审批处理",
}

STEP_LABELS = {
    "intake": "受理工单",
    "triage": "工单分诊",
    "retrieve_context": "拉取上下文",
    "propose_action": "生成动作建议",
    "approval_gate": "审批处理",
    "execute_action": "执行动作",
    "draft_reply": "生成回复",
    "finalize": "完结归档",
}

DECISION_LABELS = {
    "approve": "批准执行",
    "edit": "修改后执行",
    "reject": "驳回转人工",
}

SCENARIO_CONFIG = {
    "退款审批": {"category": "billing_refund", "missing_order": False, "refundable": True},
    "缺单号兜底": {"category": "billing_refund", "missing_order": True},
    "企业故障升级": {"category": "technical_issue", "tier": "enterprise"},
    "发货延迟处理": {"category": "delivery_issue", "missing_order": False},
}


def _runner() -> TicketFlowRunner:
    if "ticketflow_runner" not in st.session_state:
        project_root = Path(__file__).resolve().parents[2]
        st.session_state.ticketflow_runner = TicketFlowRunner.from_project_root(project_root)
    return st.session_state.ticketflow_runner


def _label(mapping: dict[str, str], value: Any) -> str:
    if value is None:
        return "-"
    return mapping.get(str(value), str(value))


def _badge(label: str, tone: str = "neutral") -> str:
    return f'<span class="tf-badge tf-badge-{tone}">{label}</span>'


def _source_badge(source: str) -> str:
    mapping = {
        "rule": ("安全兜底", "neutral"),
        "llm_cloud": ("云端模型", "accent"),
        "llm_minimind": ("本地模型", "success"),
        "human_edit": ("人工修订", "warning"),
    }
    label, tone = mapping.get(source, (source, "neutral"))
    return _badge(label, tone)


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --tf-bg: #f3f6fa;
            --tf-surface: #ffffff;
            --tf-surface-muted: #f8fafc;
            --tf-border: #dbe3ec;
            --tf-border-strong: #c8d3df;
            --tf-text: #14202b;
            --tf-muted: #5f6f82;
            --tf-accent: #1f5eff;
            --tf-accent-soft: #eaf1ff;
            --tf-success: #157347;
            --tf-warning: #b76e00;
            --tf-danger: #c92a2a;
        }

        .stApp {
            background: var(--tf-bg);
            color: var(--tf-text);
        }

        [data-testid="stHeader"] {
            background: rgba(0, 0, 0, 0);
        }

        [data-testid="stSidebar"] {
            background: #f8fafc;
            border-right: 1px solid var(--tf-border);
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
            background: var(--tf-surface);
            border: 1px solid var(--tf-border);
            border-radius: 14px;
            padding: 1rem 1.15rem;
            margin-bottom: 0.9rem;
        }

        .tf-page-title {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--tf-text);
            margin-bottom: 0.2rem;
        }

        .tf-page-subtitle {
            color: var(--tf-muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }

        .tf-panel {
            background: var(--tf-surface);
            border: 1px solid var(--tf-border);
            border-radius: 14px;
            padding: 0.95rem 1rem;
            box-shadow: none;
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
            border-radius: 10px;
            border: 1px solid var(--tf-border-strong);
            background: #ffffff;
            color: var(--tf-text);
            font-weight: 600;
            box-shadow: none;
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
    st.markdown(
        f"""
        <div class="tf-panel">
            <div class="tf-panel-title">{title}</div>
            {f'<div>{badges_html}</div>' if badges_html else ''}
            {f'<div class="tf-caption">{body}</div>' if body else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
        st.write(ticket.body)
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
            st.write(item["detail"])
            if item.get("payload"):
                with st.expander("查看节点数据", expanded=False):
                    st.json(item["payload"])


def _render_tool_calls(state: dict[str, Any]) -> None:
    tool_calls = state.get("tool_calls", [])
    if not tool_calls:
        st.info("当前工单尚未产生工具调用。")
        return
    for idx, tool_call in enumerate(tool_calls, start=1):
        with st.container(border=True):
            st.markdown(f"**{idx}. {tool_call['tool']}**")
            if tool_call.get("summary"):
                st.write(tool_call["summary"])
            if tool_call.get("args"):
                st.caption("请求参数")
                st.json(tool_call["args"])
            if tool_call.get("result") is not None:
                with st.expander("查看返回结果", expanded=False):
                    st.json(tool_call["result"])
            elif tool_call:
                with st.expander("查看原始记录", expanded=False):
                    st.json(tool_call)


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
            _badge(f"score={doc.score}", "neutral"),
        ]
        retrieval_mode = doc.metadata.get("retrieval")
        if retrieval_mode:
            badges.append(_badge(f"召回={retrieval_mode}", "neutral"))
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
            st.caption("引用: " + ", ".join(citation.source_path for citation in doc.citations))


def _render_audit_log(state: dict[str, Any]) -> None:
    audit_log = state.get("audit_log", [])
    if not audit_log:
        st.info("关键决策和执行动作会记录在这里。")
        return
    for event in audit_log:
        with st.container(border=True):
            st.markdown(f"**{_label(ACTOR_LABELS, event.actor)}** · `{event.event_type}`")
            st.caption(str(event.timestamp))
            st.write(event.detail)
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
                _badge(status, tone),
                _badge(recipient, "neutral"),
            ],
        )
        with st.expander("查看发送详情", expanded=False):
            st.json(record.model_dump(mode="json") if hasattr(record, "model_dump") else record)


def _render_triage_card(state: dict[str, Any]) -> None:
    triage = state.get("triage_result")
    if not triage:
        _panel("分诊结果", "工作流运行后，会在这里展示类别、优先级和 SLA 风险。")
        return
    badges = [
        _badge(_label(CATEGORY_LABELS, triage.category), "accent"),
        _badge(f"优先级：{_label(PRIORITY_LABELS, triage.priority)}", "warning" if triage.priority in {"high", "urgent"} else "neutral"),
        _badge("SLA 风险" if triage.sla_risk else "SLA 正常", "danger" if triage.sla_risk else "success"),
        _source_badge(getattr(triage, "decision_source", "rule")),
    ]
    _panel("分诊结果", triage.reasoning, badges)
    st.caption(f"置信度：{triage.confidence}")
    if getattr(triage, "fallback_reason", None):
        st.caption(f"回退说明：{triage.fallback_reason}")


def _render_action_card(state: dict[str, Any]) -> None:
    action = state.get("proposed_action")
    if not action:
        _panel("动作建议", "系统会在拉取证据后生成下一步动作建议。")
        return
    badges = [
        _badge(_label(ACTION_LABELS, action.action_type), "accent"),
        _badge("需要审批" if action.requires_approval else "无需审批", "danger" if action.requires_approval else "success"),
        _badge(action.suggested_tool, "neutral"),
        _source_badge(getattr(action, "decision_source", "rule")),
    ]
    _panel("动作建议", action.rationale, badges)
    if getattr(action, "fallback_reason", None):
        st.caption(f"回退说明：{action.fallback_reason}")
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
            _badge(f"状态：{execution_result['status']}", tone),
            _badge(execution_result.get("tool_name") or "无工具执行", "neutral"),
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
        st.caption(f"回退说明：{reply.fallback_reason}")
    st.markdown("**内部备注**")
    with st.container(border=True):
        st.write(reply.internal_note)
    if reply.citations:
        st.markdown("**引用证据**")
        for citation in reply.citations:
            st.caption(citation.source_path)


def _render_dashboard(runner: TicketFlowRunner, tickets: list[TicketRecord]) -> None:
    enterprise_count = sum(1 for ticket in tickets if ticket.customer_tier == "enterprise")
    refund_count = sum(1 for ticket in tickets if ticket.expected_category == "billing_refund")
    risk_count = _risk_candidate_count(runner, tickets)

    st.markdown(
        """
        <div class="tf-page-header">
            <div class="tf-page-title">工单处理平台</div>
            <div class="tf-page-subtitle">
                面向客服与 IT 服务团队的工单处理界面，支持分诊、知识检索、审批、执行、回复与外部协同。
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_cols = st.columns(4)
    metric_cols[0].metric("开放工单", len(tickets))
    metric_cols[1].metric("企业客户工单", enterprise_count)
    metric_cols[2].metric("风险关注工单", risk_count)
    metric_cols[3].metric("退款相关工单", refund_count)


def _render_runtime_mode(runner: TicketFlowRunner) -> None:
    settings = runner.settings
    st.sidebar.markdown("**运行环境**")
    st.sidebar.caption(f"模型后端：`{settings.model_backend}`")
    st.sidebar.caption(f"知识检索：{'开启' if settings.rag_enabled else '关闭'}")
    st.sidebar.caption(f"通知邮箱：{'已配置' if settings.incident_email_to else '未配置'}")
    st.sidebar.caption(f"知识运营邮箱：{'已配置' if settings.kb_ops_email_to else '未配置'}")
    if settings.model_backend == "minimind_split" and settings.minimind_split_enabled:
        st.sidebar.success("当前：本地双 LoRA 模式")
        st.sidebar.caption(f"结构化模型：{settings.minimind_structured_model}")
        st.sidebar.caption(f"结构化接口：{settings.minimind_structured_base_url}")
        st.sidebar.caption(f"回复模型：{settings.minimind_reply_model}")
        st.sidebar.caption(f"回复接口：{settings.minimind_reply_base_url}")
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
    st.set_page_config(page_title="TicketFlow 工单处理平台", layout="wide")
    _inject_styles()

    runner = _runner()
    all_tickets = runner.list_open_tickets(limit=500)

    st.sidebar.title("工作台设置")
    _render_runtime_mode(runner)
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
    st.session_state.selected_ticket_id = selected_ticket_id
    selected_ticket = ticket_map[selected_ticket_id]

    action_cols = st.columns([1.2, 1.0, 4.0])
    if action_cols[0].button("运行工作流", type="primary", use_container_width=True):
        st.session_state.last_result = runner.run_ticket(selected_ticket.ticket_id)
        st.session_state.active_ticket_for_result = selected_ticket.ticket_id
    if action_cols[1].button("刷新工单队列", use_container_width=True):
        st.rerun()
    action_cols[2].caption(f"当前选中：{selected_ticket.ticket_id}｜{selected_ticket.title}")

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
        tabs = st.tabs(["流转记录", "工具日志", "关联知识", "外部协同", "审计记录"])
        with tabs[0]:
            _render_trace(state)
        with tabs[1]:
            _render_tool_calls(state)
        with tabs[2]:
            _render_docs(state)
        with tabs[3]:
            _render_external_ops(state)
        with tabs[4]:
            _render_audit_log(state)

    with right:
        st.markdown("#### 处理结果")
        _render_triage_card(state)
        _render_action_card(state)

        if result is not None and result.interrupted:
            payload = result.interrupt_payload or {}
            proposed = ActionProposal.model_validate(payload["proposed_action"])
            st.warning("该工单当前处于审批等待状态。")
            with st.expander("查看审批负载", expanded=False):
                st.json(payload)
            with st.form("approval_form"):
                decision = st.radio(
                    "审批决定",
                    ["approve", "edit", "reject"],
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
                        decision_source="human_edit",
                    )
                submitted = st.form_submit_button("提交审批并继续")
            if submitted:
                decision_payload = ReviewDecision(
                    decision=decision,
                    edited_action=edited_action if decision == "edit" else None,
                    comment=comment,
                )
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

        _render_execution_card(state)
        _render_final_response(state)


if __name__ == "__main__":
    main()
