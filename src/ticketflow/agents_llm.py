from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .governance import get_requirement_profile, get_tool_approval_policy, infer_risk_level, infer_route_family
from .llm import OpenAICompatClient
from .models import (
    ActionRoutingDecision,
    ActionProposal,
    Citation,
    ContextSufficiencyResult,
    CustomerProfile,
    OrderRecord,
    ReplyFactCheckResult,
    RetrievedDoc,
    ReviewDecision,
    TicketRecord,
    TicketResponse,
    TriageResult,
)


ALLOWED_CATEGORIES = {"billing_refund", "delivery_issue", "technical_issue", "account_access", "general_inquiry"}
ALLOWED_PRIORITIES = {"low", "medium", "high", "urgent"}
ALLOWED_URGENCIES = {"standard", "next_business_day", "same_day", "sev1"}
ALLOWED_ACTIONS = {"refund", "escalation", "request_info", "status_update", "troubleshoot"}
ALLOWED_TARGET_STATUSES = {
    "open",
    "in_progress",
    "investigating",
    "monitoring",
    "waiting_on_customer",
    "pending_finance",
    "pending_human",
    "escalated",
}


def _extract_first_json_object(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_category(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "refund": "billing_refund",
        "billing": "billing_refund",
        "payment": "billing_refund",
        "delivery": "delivery_issue",
        "shipping": "delivery_issue",
        "shipment": "delivery_issue",
        "logistics": "delivery_issue",
        "technical": "technical_issue",
        "incident": "technical_issue",
        "outage": "technical_issue",
        "bug": "technical_issue",
        "login": "account_access",
        "access": "account_access",
        "auth": "account_access",
        "general": "general_inquiry",
        "question": "general_inquiry",
        "inquiry": "general_inquiry",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in ALLOWED_CATEGORIES else "general_inquiry"


def _normalize_priority(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "p0": "urgent",
        "critical": "urgent",
        "sev0": "urgent",
        "p1": "high",
        "major": "high",
        "sev1": "high",
        "p2": "medium",
        "normal": "medium",
        "moderate": "medium",
        "p3": "low",
        "minor": "low",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in ALLOWED_PRIORITIES else "medium"


def _normalize_urgency(value: object, priority: str) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "critical": "sev1",
        "sev0": "sev1",
        "sev1": "sev1",
        "urgent": "same_day",
        "same_day": "same_day",
        "today": "same_day",
        "next_business_day": "next_business_day",
        "next day": "next_business_day",
        "standard": "standard",
        "normal": "standard",
    }
    return aliases.get(raw, {"urgent": "sev1", "high": "same_day", "medium": "next_business_day", "low": "standard"}.get(priority, "standard"))


def _normalize_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"1", "true", "yes", "y", "required", "high"}


def _normalize_confidence(value: object, default: float = 0.65) -> float:
    if isinstance(value, (int, float)):
        value = float(value)
        return value if 0 <= value <= 1 else default
    raw = str(value or "").strip().lower()
    aliases = {"very_high": 0.95, "high": 0.85, "medium": 0.65, "low": 0.45}
    if raw in aliases:
        return aliases[raw]
    try:
        parsed = float(raw)
        return parsed if 0 <= parsed <= 1 else default
    except ValueError:
        return default


def _normalize_action_type(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "refund_request": "refund",
        "issue_refund_request": "refund",
        "escalate": "escalation",
        "create_escalation": "escalation",
        "need_info": "request_info",
        "ask_for_info": "request_info",
        "update_status": "status_update",
        "debug": "troubleshoot",
        "investigate": "troubleshoot",
    }
    normalized = aliases.get(raw, raw)
    return normalized if normalized in ALLOWED_ACTIONS else "status_update"


def _normalize_ticket_status(status: str) -> str:
    aliases = {
        "resolved": "in_progress",
        "done": "in_progress",
        "closed": "pending_human",
        "waiting_customer": "waiting_on_customer",
        "pending_review": "pending_human",
    }
    normalized = (status or "").strip().lower()
    return normalized if normalized in ALLOWED_TARGET_STATUSES else aliases.get(normalized, "in_progress")


def _default_target_status(action_type: str, category: str) -> str:
    if action_type == "refund":
        return "pending_finance"
    if action_type == "escalation":
        return "escalated"
    if action_type == "request_info":
        return "waiting_on_customer"
    if action_type == "troubleshoot":
        return "investigating"
    return "monitoring" if category == "delivery_issue" else "in_progress"


def _normalize_target_status(value: object, action_type: str, category: str) -> str:
    normalized = _normalize_ticket_status(str(value or ""))
    return normalized if normalized in ALLOWED_TARGET_STATUSES else _default_target_status(action_type, category)


def _standard_resolution_default_action(category: str) -> tuple[str, str]:
    if category in {"technical_issue", "account_access"}:
        return "troubleshoot", "investigating"
    if category == "delivery_issue":
        return "status_update", "monitoring"
    return "status_update", "in_progress"


def _repair_unnecessary_standard_request_info(
    *,
    action_type: str,
    target_status: str | None,
    rationale: str,
    triage: TriageResult,
    sufficiency_result: ContextSufficiencyResult,
) -> tuple[str, str | None, str, bool]:
    if (
        sufficiency_result.route_family == "standard_resolution"
        and sufficiency_result.sufficient
        and not sufficiency_result.missing_sources
        and action_type == "request_info"
    ):
        repaired_action, repaired_status = _standard_resolution_default_action(triage.category)
        return (
            repaired_action,
            repaired_status,
            (
                f"{rationale} Governance normalized an unnecessary request_info decision "
                "because the standard-resolution route had sufficient context and no missing sources."
            ),
            True,
        )
    return action_type, target_status, rationale, False


def _history_docs(retrieved_docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
    return [doc for doc in retrieved_docs if doc.source_type == "history"]


def _doc_support_score(doc: RetrievedDoc) -> float:
    rerank_score = doc.rerank_score
    if rerank_score is None:
        rerank_score = doc.metadata.get("rerank_score")
    try:
        return float(rerank_score if rerank_score is not None else doc.score)
    except (TypeError, ValueError):
        return float(doc.score)


def _metadata_float(doc: RetrievedDoc, key: str, default: float = 0.0) -> float:
    try:
        return float(doc.metadata.get(key, default))
    except (TypeError, ValueError):
        return default


def _history_support_score(doc: RetrievedDoc) -> float:
    """Calibrate history evidence strength from retrieval score plus rerank features.

    Cross-encoder/BM25/vector scores are good ranking signals, but their absolute
    scale is not stable enough to use directly as a business gate. We therefore
    keep the retrieval score as one signal and combine it with interpretable
    evidence features computed against the current ticket.
    """

    base_score = _doc_support_score(doc)
    feature_keys = {
        "rerank_text_overlap",
        "rerank_title_overlap",
        "rerank_metadata_overlap",
        "rerank_exact_product_match",
        "feature_rerank_score",
    }
    if not any(key in doc.metadata for key in feature_keys):
        return round(max(0.0, min(1.0, base_score)), 3)
    text_overlap = _metadata_float(doc, "rerank_text_overlap")
    title_overlap = _metadata_float(doc, "rerank_title_overlap")
    metadata_overlap = _metadata_float(doc, "rerank_metadata_overlap")
    exact_product_match = _metadata_float(doc, "rerank_exact_product_match")
    feature_score = _metadata_float(doc, "feature_rerank_score", default=base_score)
    calibrated = (
        0.30 * base_score
        + 0.20 * feature_score
        + 0.18 * text_overlap
        + 0.17 * title_overlap
        + 0.10 * exact_product_match
        + 0.05 * metadata_overlap
    )
    return round(max(0.0, min(1.0, calibrated)), 3)


def _supporting_history_docs(retrieved_docs: list[RetrievedDoc], min_history_strength: float = 0.0) -> list[RetrievedDoc]:
    docs = [doc for doc in _history_docs(retrieved_docs) if _history_support_score(doc) >= min_history_strength]
    return sorted(docs, key=_history_support_score, reverse=True)


def _supporting_history_doc_ids(retrieved_docs: list[RetrievedDoc], min_history_strength: float = 0.0) -> list[str]:
    return [f"history:{doc.doc_id}" for doc in _supporting_history_docs(retrieved_docs, min_history_strength)]


def _source_doc_ids(retrieved_docs: list[RetrievedDoc], source_type: str) -> list[str]:
    return [f"{doc.source_type}:{doc.doc_id}" for doc in retrieved_docs if doc.source_type == source_type]


def _available_sources(*, order: OrderRecord | None, policy_hits: list[RetrievedDoc], retrieved_docs: list[RetrievedDoc]) -> set[str]:
    sources = {doc.source_type for doc in retrieved_docs}
    if policy_hits:
        sources.add("policy")
    if order is not None and order.order_id:
        sources.add("order")
    return sources


def _supporting_doc_ids(
    *,
    required_sources: list[str],
    order: OrderRecord | None,
    policy_hits: list[RetrievedDoc],
    retrieved_docs: list[RetrievedDoc],
    min_history_strength: float = 0.0,
) -> list[str]:
    doc_ids: list[str] = []
    for source in required_sources:
        if source == "policy":
            doc_ids.extend(f"policy:{doc.doc_id}" for doc in policy_hits[:2])
        elif source == "order" and order is not None and order.order_id:
            doc_ids.append(f"order:{order.order_id}")
        elif source == "history":
            doc_ids.extend(_supporting_history_doc_ids(retrieved_docs, min_history_strength)[:2])
        elif source == "kb":
            doc_ids.extend(_source_doc_ids(retrieved_docs, "kb")[:2])
        elif source == "customer":
            doc_ids.extend(_source_doc_ids(retrieved_docs, "customer")[:1])
    deduped: list[str] = []
    for doc_id in doc_ids:
        if doc_id not in deduped:
            deduped.append(doc_id)
    return deduped


def _missing_required_sources(
    *,
    required_sources: list[str],
    order: OrderRecord | None,
    policy_hits: list[RetrievedDoc],
    retrieved_docs: list[RetrievedDoc],
    min_history_strength: float = 0.0,
) -> list[str]:
    available = _available_sources(order=order, policy_hits=policy_hits, retrieved_docs=retrieved_docs)
    missing = [source for source in required_sources if source not in available]
    if "history" in required_sources and "history" not in missing:
        if not _supporting_history_docs(retrieved_docs, min_history_strength):
            missing.append("history")
    return missing


def _compute_support_strength(
    *,
    required_sources: list[str],
    order: OrderRecord | None,
    policy_hits: list[RetrievedDoc],
    retrieved_docs: list[RetrievedDoc],
    min_history_strength: float = 0.0,
) -> float:
    if not required_sources:
        return 1.0

    source_strengths: list[float] = []
    for source in required_sources:
        if source == "policy":
            source_strengths.append(1.0 if policy_hits else 0.0)
        elif source == "order":
            source_strengths.append(1.0 if order is not None and order.order_id else 0.0)
        elif source == "history":
            history_docs = _supporting_history_docs(retrieved_docs, min_history_strength)
            history_strength = _history_support_score(history_docs[0]) if history_docs else 0.0
            source_strengths.append(history_strength)
        elif source == "kb":
            kb_docs = [doc for doc in retrieved_docs if doc.source_type == "kb"]
            kb_strength = max((_doc_support_score(doc) for doc in kb_docs), default=0.0)
            source_strengths.append(kb_strength)
        elif source == "customer":
            customer_docs = [doc for doc in retrieved_docs if doc.source_type == "customer"]
            customer_strength = max((_doc_support_score(doc) for doc in customer_docs), default=0.0)
            source_strengths.append(customer_strength)
    if not source_strengths:
        return 0.0
    return round(sum(source_strengths) / len(source_strengths), 3)


def _collect_top_citations(retrieved_docs: list[RetrievedDoc]) -> list[Citation]:
    return [citation for doc in retrieved_docs[:3] for citation in doc.citations[:1]]


def _extract_unsupported_claims(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in re.split(r"[;\n|]", value) if part.strip()]
    return []


def _fallback_reason(step: str, source: str, exc: Exception) -> str:
    return f"{step} 从 {source} 回退：{type(exc).__name__}"


def _triage_payload_valid(payload: dict[str, object] | None) -> bool:
    if payload is None:
        return False
    required = {"category", "priority", "urgency", "confidence", "reasoning"}
    if not required.issubset(payload):
        return False
    if str(payload.get("category") or "").strip().lower() not in ALLOWED_CATEGORIES:
        return False
    if str(payload.get("priority") or "").strip().lower() not in ALLOWED_PRIORITIES:
        return False
    if str(payload.get("urgency") or "").strip().lower() not in ALLOWED_URGENCIES:
        return False
    return True


def _sla_payload_valid(payload: dict[str, object] | None) -> bool:
    if payload is None:
        return False
    required = {"priority", "urgency", "sla_risk_score", "reasoning", "evidence"}
    if not required.issubset(payload):
        return False
    if str(payload.get("priority") or "").strip().lower() not in ALLOWED_PRIORITIES:
        return False
    if str(payload.get("urgency") or "").strip().lower() not in ALLOWED_URGENCIES:
        return False
    try:
        score = float(payload.get("sla_risk_score"))
    except (TypeError, ValueError):
        return False
    return 0 <= score <= 1


def _action_payload_valid(payload: dict[str, object] | None) -> bool:
    if payload is None:
        return False
    required = {"action_type", "target_status", "requires_approval", "rationale", "confidence"}
    if not required.issubset(payload):
        return False
    if str(payload.get("action_type") or "").strip().lower() not in ALLOWED_ACTIONS:
        return False
    if _normalize_ticket_status(str(payload.get("target_status") or "")) not in ALLOWED_TARGET_STATUSES:
        return False
    return True


def _reply_payload_valid(payload: dict[str, object] | None) -> bool:
    if payload is None:
        return False
    required = {"customer_reply", "internal_note", "status"}
    if not required.issubset(payload):
        return False
    return bool(str(payload.get("customer_reply") or "").strip()) and bool(str(payload.get("internal_note") or "").strip())


def _strip_markdown_fences(text: str) -> str:
    text = re.sub(r"^\s*```[a-zA-Z]*", "", text).strip()
    text = re.sub(r"```\s*$", "", text).strip()
    return text


def _sanitize_local_reply_text(text: str) -> str:
    cleaned = _strip_markdown_fences(text)
    cleaned = re.sub(r"</?think>", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"<\|[^>]+?\|>", " ", cleaned)
    cleaned = re.sub(r"E-s\[[^\]]*\][,，。！？!?\s]*", " ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\[[^\]]*数字[^\]]*\]", " ", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n+", "\n", cleaned).strip()
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    if lines:
        cleaned = lines[0]

    pieces = [part.strip() for part in re.split(r"[。！？!?；;]+", cleaned) if part.strip()]
    deduped: list[str] = []
    for part in pieces:
        if not deduped or deduped[-1] != part:
            deduped.append(part)
    if deduped:
        cleaned = "，".join(deduped)

    cleaned = re.sub(r"(，\s*){2,}", "，", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip("，, ")
    return cleaned


def _looks_like_bad_local_reply(text: str) -> bool:
    if not text:
        return True
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    ascii_letters = re.findall(r"[A-Za-z]", text)
    normalized = re.sub(r"\s+", "", text)
    generic_closings = {
        "请问还有其他还可以帮到您的吗",
        "请问还有什么可以帮到您的吗",
        "请问还有其他可以帮到您的吗",
    }
    if len(normalized) < 8:
        return True
    if len(chinese_chars) < 4:
        return True
    if len(ascii_letters) > max(10, len(chinese_chars) * 2):
        return True
    if normalized.strip("。！？!?，, ") in generic_closings:
        return True
    if re.search(r"(请问还有其他还可以帮到您的吗)", text) and text.count("请问还有其他还可以帮到您的吗") > 1:
        return True
    if re.search(r"^[A-Za-z][A-Za-z0-9 ,.'\":;!?-]{12,}$", text):
        return True
    return False


def _build_reply_contract(
    ticket: TicketRecord,
    triage: TriageResult,
    proposed_action: ActionProposal,
    execution_result: object,
    retrieved_docs: list[RetrievedDoc],
) -> dict[str, Any]:
    execution_status = getattr(execution_result, "status", None)
    if isinstance(execution_result, dict):
        execution_status = execution_result.get("status")
    execution_status = str(execution_status or "").strip().lower()
    action_type = proposed_action.action_type
    target_status = proposed_action.target_status
    route_family = proposed_action.route_family or "standard_resolution"
    missing_sources = set(proposed_action.missing_sources)

    supported_claims: list[str] = [
        "我们已经收到您的反馈",
        "我们已经收到您的问题",
        "后续进展会尽快同步给您",
    ]
    required_claims: list[str] = []
    forbidden_claims: list[str] = [
        "已经退款成功",
        "退款已经完成",
        "已经打款",
        "已经退回原支付账户",
        "问题已经完全修复",
        "服务已经恢复正常",
        "已经送达",
        "已经发货完成",
        "保证今天恢复",
        "保证马上解决",
    ]
    status_contract: list[str] = []
    template_reply = "我们已经收到您反馈的问题，正在根据当前信息继续处理，后续进展会尽快同步给您。"

    if action_type == "refund":
        forbidden_claims.extend(
            [
                "退款审核已经通过",
                "财务已经完成退款",
                "款项会立即到账",
            ]
        )
        if target_status == "pending_finance" or execution_status == "executed":
            template_reply = "我们已经收到您的退款相关诉求，当前工单已进入退款审核流程，后续进展会尽快同步给您。"
            supported_claims.extend(
                [
                    "我们已经收到您的退款相关诉求",
                    "当前工单已进入退款审核流程",
                    "当前处于退款审核阶段",
                    "后续进展会尽快同步给您",
                ]
            )
            required_claims.append("退款审核流程")
            status_contract.append("只能表述为已进入退款审核或待财务处理阶段，不得表述为已退款成功。")
        elif target_status == "waiting_on_customer" or "order" in missing_sources:
            template_reply = "为尽快继续处理，请您补充订单号、支付信息或相关截图，我们收到后会立即跟进。"
            supported_claims.extend(
                [
                    "为尽快继续处理",
                    "请您补充订单号",
                    "请您补充支付信息",
                    "我们收到后会立即跟进",
                ]
            )
            required_claims.append("请您补充")
            status_contract.append("只能表述为待补充退款核验信息，不得表述为已进入退款审核。")
        else:
            template_reply = "我们已经收到您的退款相关诉求，正在核验订单与支付信息，确认后会尽快同步处理进展。"
            supported_claims.extend(
                [
                    "我们已经收到您的退款相关诉求",
                    "正在核验订单与支付信息",
                    "确认后会尽快同步处理进展",
                    "后续进展会尽快同步给您",
                ]
            )
            required_claims.append("核验订单与支付信息")
            status_contract.append("只能表述为退款核验中，不得表述为审核通过、打款完成或原路退回。")
    elif action_type == "escalation":
        forbidden_claims.extend(
            [
                "故障已经恢复",
                "问题已经定位完成",
                "今晚一定恢复",
            ]
        )
        if target_status == "pending_human":
            template_reply = "当前问题已转交人工进一步处理，我们会结合现有信息持续跟进，并在有进展后尽快同步您。"
            supported_claims.extend(
                [
                    "当前问题已转交人工进一步处理",
                    "我们会结合现有信息持续跟进",
                    "有进展后尽快同步您",
                ]
            )
            required_claims.append("转交人工进一步处理")
            status_contract.append("只能表述为已转人工/待人工确认，不得表述为已修复或已恢复。")
        else:
            template_reply = "当前问题已升级给相关团队优先排查，我们会基于现有信息持续跟进，并在有进展后尽快同步您。"
            supported_claims.extend(
                [
                    "当前问题已升级给相关团队优先排查",
                    "我们会基于现有信息持续跟进",
                    "有进展后尽快同步您",
                ]
            )
            required_claims.append("升级给相关团队")
            status_contract.append("只能表述为已升级排查，不得表述为已定位完成、已修复或已恢复。")
    elif action_type == "request_info":
        requested_items: list[str] = []
        if "order" in missing_sources:
            requested_items.append("订单号")
        if "policy" in missing_sources and route_family == "refund_candidate":
            requested_items.append("支付信息")
        if "history" in missing_sources or route_family == "escalation_candidate":
            requested_items.append("受影响范围")
        if not requested_items:
            requested_items = ["订单号", "报错截图", "受影响范围"]
        request_phrase = "、".join(requested_items)
        template_reply = f"为尽快继续处理，请您补充{request_phrase}等关键信息，我们收到后会立即跟进。"
        supported_claims.extend(
            [
                "为尽快继续处理",
                "请您补充关键信息",
                "我们收到后会立即跟进",
            ]
            + [f"请您补充{item}" for item in requested_items]
        )
        required_claims.append("请您补充")
        status_contract.append("只能表述为待客户补充信息，不得表述为问题已解决或已进入高风险执行。")
    elif action_type == "status_update":
        if target_status == "pending_human":
            template_reply = "当前问题已转交人工进一步处理，我们会结合现有信息持续跟进，并在有进展后尽快同步您。"
            supported_claims.extend(
                [
                    "当前问题已转交人工进一步处理",
                    "我们会结合现有信息持续跟进",
                    "有进展后尽快同步您",
                ]
            )
            required_claims.append("转交人工进一步处理")
            status_contract.append("只能表述为已转人工处理，不得表述为已完成高风险操作。")
        else:
            template_reply = "我们正在结合当前信息继续处理，后续进展会尽快同步给您。"
            supported_claims.extend(
                [
                    "我们正在结合当前信息继续处理",
                    "后续进展会尽快同步给您",
                ]
            )
            required_claims.append("继续处理")
            status_contract.append("只能表述为处理中，不得表述为已完成。")
    elif action_type == "troubleshoot":
        template_reply = "我们已经开始排查当前问题，会优先核对报错信息、影响范围和可行的临时处理方案。"
        supported_claims.extend(
            [
                "我们已经开始排查当前问题",
                "会优先核对报错信息",
                "会确认影响范围",
            ]
        )
        required_claims.append("开始排查")
        status_contract.append("只能表述为排查中，不得表述为已修复。")
    else:
        supported_claims.extend(
            [
                "我们正在根据当前信息继续处理",
                "后续进展会尽快同步给您",
            ]
        )
        required_claims.append("继续处理")
        status_contract.append("只能表述为继续处理或持续跟进，不得表述为已完成。")

    if target_status == "waiting_on_customer":
        supported_claims.append("请您先补充相关信息")
    if target_status == "pending_human":
        supported_claims.append("当前已转交人工进一步处理")
    if ticket.customer_tier == "enterprise" and triage.priority in {"high", "urgent"} and action_type != "request_info":
        supported_claims.append("当前将按高优先级持续跟进")

    evidence_summary = [
        f"[{doc.source_type}] {doc.title}: {doc.snippet}"
        for doc in retrieved_docs[:6]
    ]
    generic_safe_phrases = [
        "您好",
        "感谢您的反馈",
        "我们已经收到",
        "后续进展会尽快同步给您",
        "持续跟进",
        "辛苦您",
    ]
    contract_summary = (
        f"route_family={route_family}; action_type={action_type}; target_status={target_status}; "
        f"execution_status={execution_status or 'none'}; status_contract={' / '.join(status_contract)}"
    )

    return {
        "template_reply": template_reply,
        "supported_claims": _dedupe_preserve_order(supported_claims),
        "required_claims": _dedupe_preserve_order(required_claims),
        "forbidden_claims": _dedupe_preserve_order(forbidden_claims),
        "generic_safe_phrases": generic_safe_phrases,
        "evidence_summary": evidence_summary,
        "status_contract": status_contract,
        "contract_summary": contract_summary,
    }


def _build_conservative_local_reply(
    ticket: TicketRecord,
    triage: TriageResult,
    proposed_action: ActionProposal,
    execution_result: object,
) -> str:
    return str(
        _build_reply_contract(
            ticket,
            triage,
            proposed_action,
            execution_result,
            [],
        )["template_reply"]
    )


def _should_use_governance_safe_reply(proposed_action: ActionProposal) -> bool:
    if proposed_action.decision_source != "governance":
        return False
    if proposed_action.sufficiency_passed:
        return False
    return proposed_action.action_type in {"request_info", "status_update"}


def _normalize_match_text(text: str) -> str:
    normalized = str(text or "").strip().lower()
    normalized = re.sub(r"\s+", "", normalized)
    normalized = re.sub(r"[，。！？；：、“”\"'`()（）【】\[\],.!?;:\-_/]", "", normalized)
    return normalized


def _split_reply_sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"[。！？；\n]+", str(text or "")) if part.strip()]


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    deduped: list[str] = []
    for item in items:
        if item and item not in deduped:
            deduped.append(item)
    return deduped


def _build_reply_claim_inventory(
    ticket: TicketRecord,
    triage: TriageResult,
    proposed_action: ActionProposal,
    execution_result: object,
    retrieved_docs: list[RetrievedDoc],
) -> dict[str, Any]:
    return _build_reply_contract(
        ticket,
        triage,
        proposed_action,
        execution_result,
        retrieved_docs,
    )


def _deterministic_reply_fact_check(
    reply_text: str,
    claim_inventory: dict[str, Any],
    *,
    threshold: float,
) -> dict[str, object]:
    normalized_reply = _normalize_match_text(reply_text)
    template_reply = _normalize_match_text(str(claim_inventory.get("template_reply") or ""))
    supported_claims = [_normalize_match_text(item) for item in claim_inventory.get("supported_claims", [])]
    required_claims = [_normalize_match_text(item) for item in claim_inventory.get("required_claims", [])]
    forbidden_claims = [_normalize_match_text(item) for item in claim_inventory.get("forbidden_claims", [])]
    generic_safe = [_normalize_match_text(item) for item in claim_inventory.get("generic_safe_phrases", [])]

    if not normalized_reply:
        return {
            "decisive": True,
            "grounded": False,
            "policy_safe": False,
            "hallucination_risk": True,
            "supported_claim_ratio": 0.0,
            "unsupported_claims": ["empty_reply"],
            "review_note": "Deterministic fact-check rejected an empty reply.",
            "review_passed": False,
            "needs_rewrite": True,
        }

    matched_forbidden = [claim for claim in forbidden_claims if claim and claim in normalized_reply]
    if matched_forbidden:
        return {
            "decisive": True,
            "grounded": False,
            "policy_safe": False,
            "hallucination_risk": True,
            "supported_claim_ratio": 0.0,
            "unsupported_claims": matched_forbidden,
            "review_note": "Deterministic fact-check found forbidden completion or commitment claims.",
            "review_passed": False,
            "needs_rewrite": True,
        }

    if template_reply and normalized_reply == template_reply:
        return {
            "decisive": True,
            "grounded": True,
            "policy_safe": True,
            "hallucination_risk": False,
            "supported_claim_ratio": 1.0,
            "unsupported_claims": [],
            "review_note": "Reply matches the deterministic conservative template for this action route.",
            "review_passed": True,
            "needs_rewrite": False,
        }

    sentences = _split_reply_sentences(reply_text)
    substantive_sentences = [sentence for sentence in sentences if _normalize_match_text(sentence)]
    if not substantive_sentences:
        substantive_sentences = [reply_text]

    unsupported_sentences: list[str] = []
    supported_hits = 0
    for sentence in substantive_sentences:
        normalized_sentence = _normalize_match_text(sentence)
        if not normalized_sentence:
            continue
        if any(token and token in normalized_sentence for token in generic_safe):
            supported_hits += 1
            continue
        if any(claim and claim in normalized_sentence for claim in supported_claims):
            supported_hits += 1
            continue
        unsupported_sentences.append(sentence.strip())

    supported_claim_ratio = round(supported_hits / max(len(substantive_sentences), 1), 3)
    required_hits = sum(1 for claim in required_claims if claim and claim in normalized_reply)
    required_ok = True if not required_claims else required_hits >= 1

    if required_ok and not unsupported_sentences and supported_claim_ratio >= threshold:
        return {
            "decisive": True,
            "grounded": True,
            "policy_safe": True,
            "hallucination_risk": False,
            "supported_claim_ratio": supported_claim_ratio,
            "unsupported_claims": [],
            "review_note": "Deterministic fact-check verified that every reply sentence stayed within the allowed supported claims.",
            "review_passed": True,
            "needs_rewrite": False,
        }

    if not required_ok and supported_claim_ratio <= 0.34:
        return {
            "decisive": True,
            "grounded": False,
            "policy_safe": True,
            "hallucination_risk": True,
            "supported_claim_ratio": supported_claim_ratio,
            "unsupported_claims": unsupported_sentences or ["missing_required_supported_claim"],
            "review_note": "Deterministic fact-check found that the reply missed route-critical supported claims.",
            "review_passed": False,
            "needs_rewrite": True,
        }

    return {
        "decisive": False,
        "grounded": False,
        "policy_safe": True,
        "hallucination_risk": False,
        "supported_claim_ratio": supported_claim_ratio,
        "unsupported_claims": unsupported_sentences,
        "review_note": "Deterministic fact-check was inconclusive; escalate to LLM verifier.",
        "review_passed": False,
        "needs_rewrite": supported_claim_ratio < threshold,
    }


def _revalidate_final_reply_against_contract(
    ticket: TicketRecord,
    triage: TriageResult,
    proposed_action: ActionProposal,
    execution_result: object,
    retrieved_docs: list[RetrievedDoc],
    response: TicketResponse,
    *,
    threshold: float,
) -> dict[str, object]:
    claim_inventory = _build_reply_claim_inventory(
        ticket,
        triage,
        proposed_action,
        execution_result,
        retrieved_docs,
    )
    return _deterministic_reply_fact_check(
        response.customer_reply,
        claim_inventory,
        threshold=threshold,
    )


def _local_triage_payload_valid(payload: dict[str, object] | None) -> bool:
    if payload is None:
        return False
    required = {"category", "priority", "urgency", "sla_risk"}
    if not required.issubset(payload):
        return False
    if _normalize_category(payload.get("category")) not in ALLOWED_CATEGORIES:
        return False
    if _normalize_priority(payload.get("priority")) not in ALLOWED_PRIORITIES:
        return False
    if _normalize_urgency(payload.get("urgency"), _normalize_priority(payload.get("priority"))) not in ALLOWED_URGENCIES:
        return False
    return True


def _local_action_payload_valid(payload: dict[str, object] | None) -> bool:
    if payload is None:
        return False
    required = {"action_type", "approval_required", "target_status"}
    if not required.issubset(payload):
        return False
    if _normalize_action_type(payload.get("action_type")) not in ALLOWED_ACTIONS:
        return False
    if _normalize_ticket_status(str(payload.get("target_status") or "")) not in ALLOWED_TARGET_STATUSES:
        return False
    return True


def _call_structured_json(
    llm_client: OpenAICompatClient,
    *,
    system_prompt: str,
    user_prompt: str,
    validator: Callable[[dict[str, object] | None], bool],
    schema_hint: str,
) -> dict[str, object]:
    if llm_client.source_label == "llm_cloud":
        payload = llm_client.chat_json(system_prompt, user_prompt)
    else:
        raw = llm_client.chat_text(system_prompt, user_prompt)
        payload = _extract_first_json_object(raw)
    if validator(payload):
        return payload or {}
    repair_prompt = (
        f"{schema_hint}\n"
        f"invalid_json={json.dumps(payload or {}, ensure_ascii=False)}\n"
        "Return JSON only."
    )
    repaired = _extract_first_json_object(
        llm_client.chat_text(
            "You repair invalid structured JSON for enterprise ticket workflows.",
            repair_prompt,
        )
    )
    if not validator(repaired):
        raise ValueError("invalid structured json after self-repair")
    return repaired or {}


@dataclass(slots=True)
class TriageAgent:
    primary_llm_client: OpenAICompatClient | None = None
    fallback_llm_client: OpenAICompatClient | None = None
    intent_recognizer: Any | None = None
    sla_risk_review_enabled: bool = True
    sla_risk_threshold: float = 0.72

    def analyze(self, ticket: TicketRecord) -> TriageResult:
        intent_hint = self._predict_intent(ticket)
        if self.primary_llm_client is None:
            if intent_hint is not None:
                return self._result_from_intent_hint(ticket, intent_hint)
            return self._result_from_ticket_fields(ticket)
        try:
            return self._analyze_with_llm(ticket, self.primary_llm_client, intent_hint)
        except Exception as exc:
            primary_reason = _fallback_reason("分诊", self.primary_llm_client.source_label, exc)
            if self.fallback_llm_client is not None:
                result = self._analyze_with_llm(ticket, self.fallback_llm_client, intent_hint)
                result.fallback_reason = primary_reason
                return result
            if intent_hint is not None:
                result = self._result_from_intent_hint(ticket, intent_hint)
                result.fallback_reason = primary_reason
                return result
            result = self._result_from_ticket_fields(ticket)
            result.fallback_reason = primary_reason
            return result

    def _predict_intent(self, ticket: TicketRecord) -> dict[str, Any] | None:
        if self.intent_recognizer is None:
            return None
        try:
            prediction = self.intent_recognizer.predict(ticket)
            if hasattr(prediction, "__dict__"):
                return {
                    "category": getattr(prediction, "category", None),
                    "confidence": getattr(prediction, "confidence", None),
                    "scores": getattr(prediction, "scores", None),
                    "model_name": getattr(prediction, "model_name", None),
                }
            return dict(prediction)
        except Exception:
            return None

    def _result_from_ticket_fields(self, ticket: TicketRecord) -> TriageResult:
        text = f"{ticket.title}\n{ticket.body}\n{ticket.product}".lower()
        category = ticket.expected_category or "general_inquiry"
        if ticket.expected_category is None:
            refund_terms = ("refund", "payment", "billing", "charge", "invoice", "退款", "扣款", "付款", "账单", "发票")
            access_terms = ("login", "password", "mfa", "verification", "access", "登录", "密码", "验证码", "账号", "访问")
            technical_terms = ("error", "unavailable", "outage", "bug", "failed", "报错", "不可用", "故障", "失败", "异常")
            delivery_terms = ("delivery", "shipment", "shipping", "物流", "配送", "快递", "签收")
            if any(term in text for term in refund_terms) or ticket.linked_order_id:
                category = "billing_refund"
            elif any(term in text for term in access_terms):
                category = "account_access"
            elif any(term in text for term in technical_terms):
                category = "technical_issue"
            elif any(term in text for term in delivery_terms):
                category = "delivery_issue"
        priority = (
            "urgent"
            if category == "technical_issue" and ticket.customer_tier == "enterprise"
            else ("high" if category in {"billing_refund", "technical_issue", "account_access"} else "medium")
        )
        sla_terms = ("sla", "sev1", "critical", "urgent", "production", "紧急", "生产", "严重", "高优", "不可用")
        sla_risk = category == "technical_issue" and (
            ticket.customer_tier == "enterprise" or any(term in text for term in sla_terms)
        )
        if sla_risk:
            priority = "urgent"
        return TriageResult(
            category=category,  # type: ignore[arg-type]
            priority=priority,  # type: ignore[arg-type]
            urgency=_normalize_urgency(priority, priority),
            sla_risk=sla_risk,
            sla_risk_score=1.0 if sla_risk else 0.0,
            confidence=0.58 if ticket.expected_category is None else 0.72,
            reasoning=f"离线评测模式下未配置 LLM/意图模型，使用工单字段生成规则分诊：{category}。",
            decision_source="rule",
            fallback_reason="No LLM client configured for triage; used deterministic ticket-field fallback.",
            sla_risk_reasoning="根据客户等级、工单类别和紧急关键词进行 SLA 风险兜底判断。",
            sla_risk_evidence=[],
        )

    def _result_from_intent_hint(self, ticket: TicketRecord, intent_hint: dict[str, Any]) -> TriageResult:
        category = _normalize_category(intent_hint.get("category"))
        text = f"{ticket.title}\n{ticket.body}".lower()
        priority = "urgent" if category == "technical_issue" and ticket.customer_tier == "enterprise" else ("high" if category in {"billing_refund", "technical_issue"} else "medium")
        sla_risk = category == "technical_issue" and (
            ticket.customer_tier == "enterprise" or any(token in text for token in ("sla", "生产", "critical", "sev1"))
        )
        if sla_risk:
            priority = "urgent"
        return TriageResult(
            category=category,  # type: ignore[arg-type]
            priority=priority,  # type: ignore[arg-type]
            urgency=_normalize_urgency(priority, priority),
            sla_risk=sla_risk,
            sla_risk_score=1.0 if sla_risk else 0.0,
            confidence=_normalize_confidence(intent_hint.get("confidence"), 0.62),
            reasoning=f"BERT 意图识别给出类别候选：{category}。",
            decision_source="rule",
            sla_risk_reasoning="BERT 意图识别结合客户等级和关键词规则完成 SLA 风险兜底。",
            sla_risk_evidence=[],
        )

    def _analyze_with_llm(self, ticket: TicketRecord, llm_client: OpenAICompatClient, intent_hint: dict[str, Any] | None = None) -> TriageResult:
        hint_text = f"\nbert_intent_hint={json.dumps(intent_hint, ensure_ascii=False)}" if intent_hint else ""
        if llm_client.source_label == "llm_minimind":
            prompt = (
                "请对下面工单做结构化分诊，并只输出 JSON。必须包含 category、priority、urgency、sla_risk。\n"
                f"channel={ticket.channel}\n"
                f"customer_tier={ticket.customer_tier}\n"
                f"product={ticket.product}\n"
                f"linked_order_id={ticket.linked_order_id or 'none'}\n"
                f"title_cn={ticket.title}\n"
                f"body_cn={ticket.body}"
                f"{hint_text}"
            )
            payload = _call_structured_json(
                llm_client,
                system_prompt="You are MiniMind-Ticket structured adapter. Only solve structured enterprise ticket tasks and always return strict JSON.",
                user_prompt=prompt,
                validator=_local_triage_payload_valid,
                schema_hint=(
                    "Required keys: category, priority, urgency, sla_risk. "
                    "Allowed category: billing_refund, delivery_issue, technical_issue, account_access, general_inquiry. "
                    "Allowed priority: low, medium, high, urgent. "
                    "Allowed urgency: standard, next_business_day, same_day, sev1."
                ),
            )
            return TriageResult(
                category=_normalize_category(payload.get("category")),
                priority=_normalize_priority(payload.get("priority")),
                urgency=_normalize_urgency(payload.get("urgency"), _normalize_priority(payload.get("priority"))),
                sla_risk=_normalize_bool(payload.get("sla_risk")),
                sla_risk_score=1.0 if _normalize_bool(payload.get("sla_risk")) else 0.0,
                confidence=0.72,
                reasoning="MiniMind 结构化适配器完成工单分诊。",
                decision_source=llm_client.source_label,  # type: ignore[assignment]
                fallback_reason=None,
                sla_risk_reasoning="MiniMind 结构化适配器直接输出 SLA 风险标签。",
                sla_risk_evidence=[],
            )

        prompt = (
            "Return JSON only with keys category, priority, urgency, confidence, reasoning. "
            "Allowed category: billing_refund, delivery_issue, technical_issue, account_access, general_inquiry. "
            "Allowed priority: low, medium, high, urgent. "
            "Allowed urgency: standard, next_business_day, same_day, sev1. "
            "Do not output SLA risk here.\n"
            f"customer_tier={ticket.customer_tier}\n"
            f"channel={ticket.channel}\n"
            f"product={ticket.product}\n"
            f"linked_order_id={ticket.linked_order_id or 'none'}\n"
            f"title_cn={ticket.title}\n"
            f"body_cn={ticket.body}"
            f"{hint_text}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are a ticket triage model.",
            user_prompt=prompt,
            validator=_triage_payload_valid,
            schema_hint=(
                "Required keys: category, priority, urgency, confidence, reasoning. "
                "Allowed category: billing_refund, delivery_issue, technical_issue, account_access, general_inquiry. "
                "Allowed priority: low, medium, high, urgent. "
                "Allowed urgency: standard, next_business_day, same_day, sev1."
            ),
        )
        result = TriageResult(
            category=_normalize_category(payload.get("category")),
            priority=_normalize_priority(payload.get("priority")),
            urgency=_normalize_urgency(payload.get("urgency"), _normalize_priority(payload.get("priority"))),
            sla_risk=False,
            sla_risk_score=0.0,
            confidence=_normalize_confidence(payload.get("confidence"), 0.65),
            reasoning=str(payload.get("reasoning") or "LLM completed ticket triage."),
            decision_source=llm_client.source_label,  # type: ignore[assignment]
            fallback_reason=None,
        )
        if self.sla_risk_review_enabled:
            review = self._review_priority_and_sla_risk(ticket, result, llm_client)
            result.sla_risk_score = _normalize_confidence(review.get("sla_risk_score"), 0.0)
            result.sla_risk = result.sla_risk_score >= self.sla_risk_threshold
            result.priority = "urgent" if result.sla_risk else _normalize_priority(review.get("priority") or result.priority)
            result.urgency = "sev1" if result.priority == "urgent" else _normalize_urgency(review.get("urgency"), result.priority)
            result.sla_risk_reasoning = str(review.get("reasoning") or "LLM completed SLA risk review.")
            result.sla_risk_evidence = [str(item) for item in (review.get("evidence") or [])][:3]
        return result

    def _review_priority_and_sla_risk(
        self,
        ticket: TicketRecord,
        preliminary: TriageResult,
        llm_client: OpenAICompatClient,
    ) -> dict[str, object]:
        prompt = (
            "Return JSON only with keys priority, urgency, sla_risk_score, reasoning, evidence. "
            "Allowed priority: low, medium, high, urgent. "
            "Allowed urgency: standard, next_business_day, same_day, sev1. "
            "sla_risk_score must be between 0 and 1. "
            "Score >= 0.80 only when the text explicitly states SLA risk, contractual deadline breach, or production impact together with service-commitment language.\n"
            f"preliminary={preliminary.model_dump(mode='json')}\n"
            f"linked_order_id={ticket.linked_order_id or 'none'}\n"
            f"title_cn={ticket.title}\n"
            f"body_cn={ticket.body}"
        )
        return _call_structured_json(
            llm_client,
            system_prompt="You are the SLA-risk review model in an enterprise ticket workflow.",
            user_prompt=prompt,
            validator=_sla_payload_valid,
            schema_hint=(
                "Required keys: priority, urgency, sla_risk_score, reasoning, evidence. "
                "Allowed priority: low, medium, high, urgent. "
                "Allowed urgency: standard, next_business_day, same_day, sev1. "
                "sla_risk_score must be between 0 and 1."
            ),
        )


@dataclass(slots=True)
class KnowledgeAgent:
    def retrieve(self, customer: CustomerProfile | None, order: OrderRecord | None, rag_docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        docs = list(rag_docs)
        if customer is not None:
            docs.append(
                RetrievedDoc(
                    doc_id=f"customer:{customer.customer_id}",
                    source_type="customer",
                    title=f"{customer.name} 的客户画像",
                    snippet=f"客户等级={customer.customer_tier}，地区={customer.region}，合作年限={customer.loyalty_years}，未关闭工单={customer.open_tickets}",
                    score=1.0,
                    metadata=customer.model_dump(mode="json"),
                    citations=[Citation(source_path=f"db:customers/{customer.customer_id}", note="客户画像")],
                )
            )
        if order is not None:
            docs.append(
                RetrievedDoc(
                    doc_id=f"order:{order.order_id}",
                    source_type="order",
                    title=f"订单 {order.order_id} 状态",
                    snippet=f"状态={order.status}，签收天数={order.delivered_days_ago}，可退款={order.eligible_for_refund}，金额={order.amount}",
                    score=1.0,
                    metadata=order.model_dump(mode="json"),
                    citations=[Citation(source_path=f"db:orders/{order.order_id}", note="订单记录")],
                )
            )
        return docs


@dataclass(slots=True)
class ResolutionAgent:
    action_primary_llm_client: OpenAICompatClient | None = None
    action_fallback_llm_client: OpenAICompatClient | None = None
    draft_primary_llm_client: OpenAICompatClient | None = None
    draft_fallback_llm_client: OpenAICompatClient | None = None
    reply_review_enabled: bool = True
    reply_supported_claim_threshold: float = 0.7

    def _build_action_proposal(
        self,
        *,
        action_type: str,
        ticket: TicketRecord,
        triage: TriageResult,
        order: OrderRecord | None,
        rationale: str,
        confidence: float,
        target_status: str | None = None,
        requires_approval: bool | None = None,
        decision_source: str = "llm_cloud",
        fallback_reason: str | None = None,
    ) -> ActionProposal:
        resolved_target_status = _normalize_target_status(target_status, action_type, triage.category)
        if action_type == "refund":
            tool_name, tool_args = "issue_refund_request", {
                "ticket_id": ticket.ticket_id,
                "order_id": order.order_id if order else ticket.linked_order_id,
                "amount": order.amount if order else None,
                "rationale": "订单满足退款条件，进入退款审批流程。",
            }
        elif action_type == "escalation":
            tool_name, tool_args = "create_escalation", {
                "ticket_id": ticket.ticket_id,
                "reason": "当前问题需要专家团队继续处理。",
                "priority": triage.priority,
            }
        else:
            tool_name, tool_args = "update_ticket_status", {"ticket_id": ticket.ticket_id, "status": resolved_target_status}
        tool_policy = get_tool_approval_policy(tool_name)
        if tool_policy.sensitive:
            resolved_requires_approval = True
        else:
            resolved_requires_approval = bool(requires_approval) if requires_approval is not None else action_type in {"refund", "escalation"}
        return ActionProposal(
            action_type=action_type,  # type: ignore[arg-type]
            target_status=resolved_target_status,
            rationale=rationale,
            requires_approval=resolved_requires_approval,
            suggested_tool=tool_name,
            tool_args=tool_args,
            confidence=round(confidence, 2),
            decision_source=decision_source,  # type: ignore[arg-type]
            fallback_reason=fallback_reason,
        )

    def propose_action(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ActionProposal:
        if self.action_primary_llm_client is None:
            return self._build_deterministic_action_fallback(
                ticket=ticket,
                triage=triage,
                order=order,
                sufficiency_result=sufficiency_result,
                action_route=action_route,
                fallback_reason="No LLM client configured for action proposal; used deterministic constrained fallback",
            )
        try:
            return self._propose_action_with_llm(self.action_primary_llm_client, ticket, triage, customer, order, policy_hits, retrieved_docs)
        except Exception as exc:
            primary_reason = _fallback_reason("动作建议", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                result = self._propose_action_with_llm(self.action_fallback_llm_client, ticket, triage, customer, order, policy_hits, retrieved_docs)
                result.fallback_reason = primary_reason
                return result
            raise RuntimeError(primary_reason) from exc

    def _propose_action_with_llm(
        self,
        llm_client: OpenAICompatClient,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ActionProposal:
        evidence_summary = _describe_action_evidence(
            ticket=ticket,
            triage=triage,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        if llm_client.source_label == "llm_minimind":
            prompt = (
                "请基于工单和分诊结果给出下一步处理动作，并只输出 JSON。必须包含 action_type、approval_required、target_status。\n"
                f"channel={ticket.channel}\n"
                f"customer_tier={ticket.customer_tier}\n"
                f"product={ticket.product}\n"
                f"linked_order_id={ticket.linked_order_id or 'none'}\n"
                f"title_cn={ticket.title}\n"
                f"body_cn={ticket.body}\n"
                f"triage={json.dumps({'category': triage.category, 'priority': triage.priority, 'sla_risk': triage.sla_risk}, ensure_ascii=False)}\n"
                f"evidence_summary={json.dumps(evidence_summary, ensure_ascii=False)}\n"
                "If required policy/history/order evidence is missing for a high-risk action, prefer request_info or pending_human."
            )
            payload = _call_structured_json(
                llm_client,
                system_prompt="You are MiniMind-Ticket structured adapter. Only solve structured enterprise ticket tasks and always return strict JSON.",
                user_prompt=prompt,
                validator=_local_action_payload_valid,
                schema_hint=(
                    "Required keys: action_type, approval_required, target_status. "
                    "Allowed action_type: refund, escalation, request_info, status_update, troubleshoot. "
                    "Allowed target_status: in_progress, investigating, monitoring, waiting_on_customer, pending_finance, pending_human, escalated."
                ),
            )
            action_type = _normalize_action_type(payload.get("action_type"))
            action_type, override_status, override_approval, override_reason = _conservative_action_override(
                action_type=action_type,
                ticket=ticket,
                triage=triage,
                order=order,
                policy_hits=policy_hits,
                retrieved_docs=retrieved_docs,
            )
            rationale = "MiniMind 结构化适配器生成动作建议。"
            if override_reason:
                rationale = f"{rationale} Conservative fallback applied: {override_reason}."
            return self._build_action_proposal(
                action_type=action_type,
                ticket=ticket,
                triage=triage,
                order=order,
                rationale=rationale,
                confidence=0.72,
                target_status=override_status or _normalize_target_status(payload.get("target_status"), action_type, triage.category),
                requires_approval=override_approval if override_approval is not None else _normalize_bool(payload.get("approval_required")),
                decision_source=llm_client.source_label,
            )

        evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:6])
        prompt = (
            "Return JSON only with keys action_type, target_status, requires_approval, rationale, confidence. "
            "Allowed action_type: refund, escalation, request_info, status_update, troubleshoot. "
            "Allowed target_status: in_progress, investigating, monitoring, waiting_on_customer, pending_finance, pending_human, escalated. "
            "If refund evidence is incomplete or no valid order exists, prefer request_info instead of refund. "
            "If policy or history evidence required for a high-risk action is missing, do not choose refund or escalation; prefer request_info or pending_human. "
            "For normal account_access issues, prefer troubleshoot instead of escalation. "
            f"channel={ticket.channel}\ncustomer_tier={ticket.customer_tier}\nproduct={ticket.product}\n"
            f"linked_order_id={ticket.linked_order_id or 'none'}\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"customer={customer.model_dump(mode='json') if customer else 'null'}\n"
            f"order={order.model_dump(mode='json') if order else 'null'}\n"
            f"policies={[doc.snippet for doc in policy_hits[:5]]}\n"
            f"evidence_summary={json.dumps(evidence_summary, ensure_ascii=False)}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are an enterprise ticket action-decision model.",
            user_prompt=prompt,
            validator=_action_payload_valid,
            schema_hint=(
                "Required keys: action_type, target_status, requires_approval, rationale, confidence. "
                "Allowed action_type: refund, escalation, request_info, status_update, troubleshoot. "
                "Allowed target_status: in_progress, investigating, monitoring, waiting_on_customer, pending_finance, pending_human, escalated."
            ),
        )
        action_type = _normalize_action_type(payload.get("action_type"))
        action_type, override_status, override_approval, override_reason = _conservative_action_override(
            action_type=action_type,
            ticket=ticket,
            triage=triage,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        rationale = str(payload.get("rationale") or "LLM generated the next action proposal.")
        if override_reason:
            rationale = f"{rationale} Conservative fallback applied: {override_reason}."
        return self._build_action_proposal(
            action_type=action_type,
            ticket=ticket,
            triage=triage,
            order=order,
            rationale=rationale,
            confidence=_normalize_confidence(payload.get("confidence"), 0.68),
            target_status=override_status or _normalize_target_status(payload.get("target_status"), action_type, triage.category),
            requires_approval=override_approval if override_approval is not None else _normalize_bool(payload.get("requires_approval")),
            decision_source=llm_client.source_label,
        )

    def draft_reply(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        proposed_action: ActionProposal,
        execution_result: object,
        retrieved_docs: list[RetrievedDoc],
        templates: list[str],
        review_decision: ReviewDecision | None = None,
    ) -> TicketResponse:
        if self.draft_primary_llm_client is None:
            raise RuntimeError("No LLM client configured for reply drafting")
        try:
            generated = self._draft_with_llm(self.draft_primary_llm_client, ticket, triage, proposed_action, execution_result, retrieved_docs, review_decision)
            if not self.reply_review_enabled:
                return generated
            if self.draft_primary_llm_client.source_label == "llm_minimind" and self.draft_fallback_llm_client is None:
                generated.review_passed = bool(generated.customer_reply.strip())
                generated.review_grounded = False
                generated.review_policy_safe = True
                generated.review_hallucination_risk = False
                generated.review_supported_claim_ratio = 0.0
                generated.review_note = "本地回复模型未启用独立校验器，当前仅完成接入级验证。"
                return generated
            verifier_client = self.draft_fallback_llm_client or self.draft_primary_llm_client
            review_payload = self._review_reply_with_llm(verifier_client, ticket, triage, proposed_action, execution_result, retrieved_docs, generated)
            if not review_payload["review_passed"] and self.draft_primary_llm_client.source_label == "llm_cloud":
                generated = self._repair_reply_with_llm(self.draft_primary_llm_client, ticket, triage, proposed_action, execution_result, retrieved_docs, generated, review_payload)
                review_payload = self._review_reply_with_llm(verifier_client, ticket, triage, proposed_action, execution_result, retrieved_docs, generated)
            elif not review_payload["review_passed"] and self.draft_fallback_llm_client is not None:
                raise ValueError("Reply adapter failed review")
            generated.review_passed = bool(review_payload["review_passed"])
            generated.review_grounded = bool(review_payload["grounded"])
            generated.review_policy_safe = bool(review_payload["policy_safe"])
            generated.review_hallucination_risk = bool(review_payload["hallucination_risk"])
            generated.review_supported_claim_ratio = float(review_payload["supported_claim_ratio"])
            generated.review_note = str(review_payload["review_note"])
            return generated
        except Exception as exc:
            primary_reason = _fallback_reason("回复生成", self.draft_primary_llm_client.source_label, exc)
            if self.draft_fallback_llm_client is not None:
                result = self._draft_with_llm(self.draft_fallback_llm_client, ticket, triage, proposed_action, execution_result, retrieved_docs, review_decision)
                result.fallback_reason = primary_reason
                return result
            raise RuntimeError(primary_reason) from exc

    def _draft_with_llm(
        self,
        llm_client: OpenAICompatClient,
        ticket: TicketRecord,
        triage: TriageResult,
        proposed_action: ActionProposal,
        execution_result: object,
        retrieved_docs: list[RetrievedDoc],
        review_decision: ReviewDecision | None,
    ) -> TicketResponse:
        claim_inventory = _build_reply_claim_inventory(
            ticket,
            triage,
            proposed_action,
            execution_result,
            retrieved_docs,
        )
        supported_claims = claim_inventory["supported_claims"]
        required_claims = claim_inventory["required_claims"]
        forbidden_claims = claim_inventory["forbidden_claims"]
        evidence_summary = claim_inventory["evidence_summary"]
        contract_summary = claim_inventory["contract_summary"]
        status_contract = claim_inventory["status_contract"]
        template_reply = claim_inventory["template_reply"]
        if llm_client.source_label == "llm_minimind":
            execution_payload = getattr(execution_result, "__dict__", execution_result)
            evidence = "\n".join(f"- {line}" for line in evidence_summary[:4])
            prompt = (
                "下面是一张企业工单的处理信息，请生成一条发给客户的中文客服回复。"
                "要求礼貌、简洁、保守，不要输出 JSON，不要承诺尚未执行的退款、修复或发货结果。"
                "回复必须满足 reply contract，只能复述 supported_claims 和 evidence 中已经支持的事实。\n"
                f"title_cn={ticket.title}\n"
                f"body_cn={ticket.body}\n"
                f"action_type={proposed_action.action_type}\n"
                f"target_status={proposed_action.target_status}\n"
                f"execution={execution_payload}\n"
                f"contract_summary={contract_summary}\n"
                f"status_contract={json.dumps(status_contract, ensure_ascii=False)}\n"
                f"supported_claims={json.dumps(supported_claims, ensure_ascii=False)}\n"
                f"required_claims={json.dumps(required_claims, ensure_ascii=False)}\n"
                f"forbidden_claims={json.dumps(forbidden_claims, ensure_ascii=False)}\n"
                f"safe_template={template_reply}\n"
                f"evidence=\n{evidence or '- none'}"
            )
            reply_text = llm_client.chat_text(
                "你是中文客服回复模型。请严格遵守 reply contract：只说 supported_claims 中被 evidence 支持的内容；如果不确定，就靠近 safe_template 的保守表达。",
                prompt,
            ).strip()
            reply_text = _sanitize_local_reply_text(reply_text)
            used_conservative_fallback = False
            if _looks_like_bad_local_reply(reply_text):
                reply_text = _build_conservative_local_reply(ticket, triage, proposed_action, execution_result)
                used_conservative_fallback = True
            return TicketResponse(
                customer_reply=reply_text or "我们已经收到您的问题，会尽快为您核实处理。",
                internal_note=proposed_action.rationale or "本地回复模型已生成客户沟通文案。",
                status=proposed_action.target_status,
                citations=_collect_top_citations(retrieved_docs),
                draft_source=llm_client.source_label,  # type: ignore[arg-type]
                fallback_reason=None,
            )

        execution_payload = getattr(execution_result, "__dict__", execution_result)
        evidence = "\n".join(f"- {line}" for line in evidence_summary[:6])
        prompt = (
            "Return JSON only with keys customer_reply, internal_note, status. "
            "Write customer_reply and internal_note in Chinese. "
            "Never claim a refund, fix, or delivery completion unless execution.status is executed. "
            "Only use supported_claims, required_claims, status_contract, and evidence_summary as factual material.\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"action={proposed_action.model_dump(mode='json')}\n"
            f"execution={execution_payload}\n"
            f"review={review_decision.model_dump(mode='json') if review_decision else 'null'}\n"
            f"contract_summary={contract_summary}\n"
            f"status_contract={json.dumps(status_contract, ensure_ascii=False)}\n"
            f"supported_claims={json.dumps(supported_claims, ensure_ascii=False)}\n"
            f"required_claims={json.dumps(required_claims, ensure_ascii=False)}\n"
            f"forbidden_claims={json.dumps(forbidden_claims, ensure_ascii=False)}\n"
            f"safe_template={template_reply}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are the reply-generation model in an enterprise ticket workflow.",
            user_prompt=prompt,
            validator=_reply_payload_valid,
            schema_hint="Required keys: customer_reply, internal_note, status.",
        )
        response = TicketResponse(
            customer_reply=str(payload.get("customer_reply") or ""),
            internal_note=str(payload.get("internal_note") or "LLM generated a structured reply."),
            status=_normalize_ticket_status(str(payload.get("status") or "in_progress")),
            citations=_collect_top_citations(retrieved_docs),
            draft_source=llm_client.source_label,  # type: ignore[arg-type]
            fallback_reason=None,
        )
        return response

    def _review_reply_with_llm(
        self,
        llm_client: OpenAICompatClient,
        ticket: TicketRecord,
        triage: TriageResult,
        proposed_action: ActionProposal,
        execution_result: object,
        retrieved_docs: list[RetrievedDoc],
        response: TicketResponse,
    ) -> dict[str, object]:
        execution_payload = getattr(execution_result, "__dict__", execution_result)
        evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:6])
        prompt = (
            "Return JSON only with keys grounded, policy_safe, hallucination_risk, supported_claim_ratio, review_note, needs_rewrite. "
            "supported_claim_ratio must be between 0 and 1. "
            f"ticket={ticket.model_dump(mode='json')}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"action={proposed_action.model_dump(mode='json')}\n"
            f"execution={execution_payload}\n"
            f"reply={response.model_dump(mode='json')}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are a grounded-reply verifier in an enterprise ticket workflow.",
            user_prompt=prompt,
            validator=lambda value: value is not None
            and {"grounded", "policy_safe", "hallucination_risk", "supported_claim_ratio", "review_note", "needs_rewrite"}.issubset(value),
            schema_hint="Required keys: grounded, policy_safe, hallucination_risk, supported_claim_ratio, review_note, needs_rewrite.",
        )
        grounded = _normalize_bool(payload.get("grounded"))
        policy_safe = _normalize_bool(payload.get("policy_safe"))
        hallucination_risk = _normalize_bool(payload.get("hallucination_risk"))
        supported_claim_ratio = _normalize_confidence(payload.get("supported_claim_ratio"), 0.0)
        needs_rewrite = _normalize_bool(payload.get("needs_rewrite")) or supported_claim_ratio < self.reply_supported_claim_threshold
        return {
            "grounded": grounded,
            "policy_safe": policy_safe,
            "hallucination_risk": hallucination_risk,
            "supported_claim_ratio": supported_claim_ratio,
            "review_note": str(payload.get("review_note") or "LLM completed reply verification."),
            "review_passed": grounded and policy_safe and not hallucination_risk and supported_claim_ratio >= self.reply_supported_claim_threshold,
            "needs_rewrite": needs_rewrite,
        }

    def _repair_reply_with_llm(
        self,
        llm_client: OpenAICompatClient,
        ticket: TicketRecord,
        triage: TriageResult,
        proposed_action: ActionProposal,
        execution_result: object,
        retrieved_docs: list[RetrievedDoc],
        response: TicketResponse,
        review_payload: dict[str, object],
    ) -> TicketResponse:
        execution_payload = getattr(execution_result, "__dict__", execution_result)
        evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:6])
        prompt = (
            "Return JSON only with keys customer_reply, internal_note, status. "
            "Rewrite the reply to remove unsupported claims and keep it conservative and evidence-grounded.\n"
            f"ticket={ticket.model_dump(mode='json')}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"action={proposed_action.model_dump(mode='json')}\n"
            f"execution={execution_payload}\n"
            f"original_reply={response.model_dump(mode='json')}\n"
            f"review_note={review_payload.get('review_note')}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are a reply-repair model in an enterprise ticket workflow.",
            user_prompt=prompt,
            validator=_reply_payload_valid,
            schema_hint="Required keys: customer_reply, internal_note, status.",
        )
        return TicketResponse(
            customer_reply=str(payload.get("customer_reply") or response.customer_reply),
            internal_note=str(payload.get("internal_note") or response.internal_note),
            status=_normalize_ticket_status(str(payload.get("status") or response.status)),
            citations=response.citations,
            draft_source=response.draft_source,
            fallback_reason=response.fallback_reason,
        )

    def _build_deterministic_sufficiency_result_v3(
        self,
        *,
        route_family: str,
        risk_level: str,
        required_sources: list[str],
        candidate_supporting_doc_ids: list[str],
        supporting_history_doc_ids: list[str],
        missing_sources: list[str],
        support_strength: float,
        fallback_reason: str | None = None,
    ) -> ContextSufficiencyResult:
        sufficient = not required_sources or not missing_sources
        fallback_route = None
        if not sufficient and route_family == "refund_candidate":
            fallback_route = "request_info"
        elif not sufficient and route_family == "escalation_candidate":
            fallback_route = "status_update"
        reasoning = (
            "Deterministic sufficiency validator found all required evidence."
            if sufficient
            else f"Deterministic sufficiency validator found missing sources: {', '.join(missing_sources)}."
        )
        return ContextSufficiencyResult(
            route_family=route_family,  # type: ignore[arg-type]
            risk_level=risk_level,  # type: ignore[arg-type]
            required_sources=required_sources,
            supporting_doc_ids=candidate_supporting_doc_ids,
            supporting_history_doc_ids=supporting_history_doc_ids,
            missing_sources=missing_sources,
            sufficient=sufficient,
            fallback_route=fallback_route,  # type: ignore[arg-type]
            support_strength=support_strength,
            reasoning=reasoning,
            decision_source="governance",
            fallback_reason=fallback_reason,
        )

    def assess_sufficiency(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ContextSufficiencyResult:
        route_family = infer_route_family(ticket, triage)
        risk_level = infer_risk_level(ticket, triage)
        requirement_profile = get_requirement_profile(route_family)
        deterministic_missing = _missing_required_sources(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        candidate_supporting_doc_ids = _supporting_doc_ids(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )

        if self.action_primary_llm_client is None:
            return self._build_deterministic_sufficiency_result_v3(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason="No LLM client configured for context sufficiency; used deterministic validator.",
            )

        try:
            return self._assess_sufficiency_with_llm(
                self.action_primary_llm_client,
                ticket,
                triage,
                customer,
                order,
                policy_hits,
                retrieved_docs,
            )
        except Exception as exc:
            primary_reason = _fallback_reason("上下文充分性判断", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                try:
                    result = self._assess_sufficiency_with_llm(
                        self.action_fallback_llm_client,
                        ticket,
                        triage,
                        customer,
                        order,
                        policy_hits,
                        retrieved_docs,
                    )
                    result.fallback_reason = primary_reason
                    return result
                except Exception as fallback_exc:
                    fallback_reason = (
                        f"{primary_reason}; "
                        f"{_fallback_reason('上下文充分性判断', self.action_fallback_llm_client.source_label, fallback_exc)}; "
                        "used deterministic validator"
                    )
                    return self._build_deterministic_sufficiency_result_v3(
                        route_family=route_family,
                        risk_level=risk_level,
                        required_sources=requirement_profile.required_sources,
                        candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                        missing_sources=deterministic_missing,
                        fallback_reason=fallback_reason,
                    )
            return self._build_deterministic_sufficiency_result_v3(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason=f"{primary_reason}; used deterministic validator",
            )

    def _build_deterministic_sufficiency_result_v2(
        self,
        *,
        route_family: str,
        risk_level: str,
        required_sources: list[str],
        candidate_supporting_doc_ids: list[str],
        missing_sources: list[str],
        fallback_reason: str | None = None,
    ) -> ContextSufficiencyResult:
        sufficient = not required_sources or not missing_sources
        fallback_route = None
        if not sufficient and route_family == "refund_candidate":
            fallback_route = "request_info"
        elif not sufficient and route_family == "escalation_candidate":
            fallback_route = "status_update"
        reasoning = (
            "Deterministic sufficiency validator found all required evidence."
            if sufficient
            else f"Deterministic sufficiency validator found missing sources: {', '.join(missing_sources)}."
        )
        return ContextSufficiencyResult(
            route_family=route_family,  # type: ignore[arg-type]
            risk_level=risk_level,  # type: ignore[arg-type]
            required_sources=required_sources,
            supporting_doc_ids=candidate_supporting_doc_ids,
            missing_sources=missing_sources,
            sufficient=sufficient,
            fallback_route=fallback_route,  # type: ignore[arg-type]
            reasoning=reasoning,
            decision_source="governance",
            fallback_reason=fallback_reason,
        )

    def assess_sufficiency(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ContextSufficiencyResult:
        route_family = infer_route_family(ticket, triage)
        risk_level = infer_risk_level(ticket, triage)
        requirement_profile = get_requirement_profile(route_family)
        deterministic_missing = _missing_required_sources(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        candidate_supporting_doc_ids = _supporting_doc_ids(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )

        if self.action_primary_llm_client is None:
            return self._build_deterministic_sufficiency_result_v2(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason="No LLM client configured for context sufficiency; used deterministic validator.",
            )

        try:
            return self._assess_sufficiency_with_llm(
                self.action_primary_llm_client,
                ticket,
                triage,
                customer,
                order,
                policy_hits,
                retrieved_docs,
            )
        except Exception as exc:
            primary_reason = _fallback_reason("上下文充分性判断", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                try:
                    result = self._assess_sufficiency_with_llm(
                        self.action_fallback_llm_client,
                        ticket,
                        triage,
                        customer,
                        order,
                        policy_hits,
                        retrieved_docs,
                    )
                    result.fallback_reason = primary_reason
                    return result
                except Exception as fallback_exc:
                    fallback_reason = (
                        f"{primary_reason}; "
                        f"{_fallback_reason('上下文充分性判断', self.action_fallback_llm_client.source_label, fallback_exc)}; "
                        "used deterministic validator"
                    )
                    return self._build_deterministic_sufficiency_result_v2(
                        route_family=route_family,
                        risk_level=risk_level,
                        required_sources=requirement_profile.required_sources,
                        candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                        missing_sources=deterministic_missing,
                        fallback_reason=fallback_reason,
                    )
            return self._build_deterministic_sufficiency_result_v2(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason=f"{primary_reason}; used deterministic validator",
            )

    def _build_deterministic_sufficiency_result(
        self,
        *,
        route_family: str,
        risk_level: str,
        required_sources: list[str],
        candidate_supporting_doc_ids: list[str],
        missing_sources: list[str],
        fallback_reason: str | None = None,
    ) -> ContextSufficiencyResult:
        sufficient = not required_sources or not missing_sources
        fallback_route = None
        if not sufficient and route_family == "refund_candidate":
            fallback_route = "request_info"
        elif not sufficient and route_family == "escalation_candidate":
            fallback_route = "status_update"
        reasoning = (
            "Deterministic sufficiency validator found all required evidence."
            if sufficient
            else f"Deterministic sufficiency validator found missing sources: {', '.join(missing_sources)}."
        )
        return ContextSufficiencyResult(
            route_family=route_family,  # type: ignore[arg-type]
            risk_level=risk_level,  # type: ignore[arg-type]
            required_sources=required_sources,
            supporting_doc_ids=candidate_supporting_doc_ids,
            missing_sources=missing_sources,
            sufficient=sufficient,
            fallback_route=fallback_route,  # type: ignore[arg-type]
            reasoning=reasoning,
            decision_source="governance",
            fallback_reason=fallback_reason,
        )

    def assess_sufficiency(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ContextSufficiencyResult:
        route_family = infer_route_family(ticket, triage)
        risk_level = infer_risk_level(ticket, triage)
        requirement_profile = get_requirement_profile(route_family)
        deterministic_missing = _missing_required_sources(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        candidate_supporting_doc_ids = _supporting_doc_ids(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )

        if self.action_primary_llm_client is None:
            return self._build_deterministic_sufficiency_result(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason="No LLM client configured for context sufficiency; used deterministic validator.",
            )

        try:
            return self._assess_sufficiency_with_llm(
                self.action_primary_llm_client,
                ticket,
                triage,
                customer,
                order,
                policy_hits,
                retrieved_docs,
            )
        except Exception as exc:
            primary_reason = _fallback_reason("上下文充分性判断", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                try:
                    result = self._assess_sufficiency_with_llm(
                        self.action_fallback_llm_client,
                        ticket,
                        triage,
                        customer,
                        order,
                        policy_hits,
                        retrieved_docs,
                    )
                    result.fallback_reason = primary_reason
                    return result
                except Exception as fallback_exc:
                    fallback_reason = (
                        f"{primary_reason}; "
                        f"{_fallback_reason('上下文充分性判断', self.action_fallback_llm_client.source_label, fallback_exc)}; "
                        "used deterministic validator"
                    )
                    return self._build_deterministic_sufficiency_result(
                        route_family=route_family,
                        risk_level=risk_level,
                        required_sources=requirement_profile.required_sources,
                        candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                        missing_sources=deterministic_missing,
                        fallback_reason=fallback_reason,
                    )
            return self._build_deterministic_sufficiency_result(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason=f"{primary_reason}; used deterministic validator",
            )

    def _build_deterministic_sufficiency_result(
        self,
        *,
        route_family: str,
        risk_level: str,
        required_sources: list[str],
        candidate_supporting_doc_ids: list[str],
        missing_sources: list[str],
        fallback_reason: str | None = None,
    ) -> ContextSufficiencyResult:
        sufficient = not required_sources or not missing_sources
        fallback_route = None
        if not sufficient and route_family == "refund_candidate":
            fallback_route = "request_info"
        elif not sufficient and route_family == "escalation_candidate":
            fallback_route = "status_update"
        reasoning = (
            "Deterministic sufficiency validator found all required evidence."
            if sufficient
            else f"Deterministic sufficiency validator found missing sources: {', '.join(missing_sources)}."
        )
        return ContextSufficiencyResult(
            route_family=route_family,  # type: ignore[arg-type]
            risk_level=risk_level,  # type: ignore[arg-type]
            required_sources=required_sources,
            supporting_doc_ids=candidate_supporting_doc_ids,
            missing_sources=missing_sources,
            sufficient=sufficient,
            fallback_route=fallback_route,  # type: ignore[arg-type]
            reasoning=reasoning,
            decision_source="governance",
            fallback_reason=fallback_reason,
        )

    def assess_sufficiency(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ContextSufficiencyResult:
        route_family = infer_route_family(ticket, triage)
        risk_level = infer_risk_level(ticket, triage)
        requirement_profile = get_requirement_profile(route_family)
        deterministic_missing = _missing_required_sources(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        candidate_supporting_doc_ids = _supporting_doc_ids(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )

        if self.action_primary_llm_client is None:
            return self._build_deterministic_sufficiency_result(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason="No LLM client configured for context sufficiency; used deterministic validator.",
            )

        try:
            return self._assess_sufficiency_with_llm(
                self.action_primary_llm_client,
                ticket,
                triage,
                customer,
                order,
                policy_hits,
                retrieved_docs,
            )
        except Exception as exc:
            primary_reason = _fallback_reason("上下文充分性判断", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                try:
                    result = self._assess_sufficiency_with_llm(
                        self.action_fallback_llm_client,
                        ticket,
                        triage,
                        customer,
                        order,
                        policy_hits,
                        retrieved_docs,
                    )
                    result.fallback_reason = primary_reason
                    return result
                except Exception as fallback_exc:
                    fallback_reason = (
                        f"{primary_reason}; "
                        f"{_fallback_reason('上下文充分性判断', self.action_fallback_llm_client.source_label, fallback_exc)}; "
                        "used deterministic validator"
                    )
                    return self._build_deterministic_sufficiency_result(
                        route_family=route_family,
                        risk_level=risk_level,
                        required_sources=requirement_profile.required_sources,
                        candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                        missing_sources=deterministic_missing,
                        fallback_reason=fallback_reason,
                    )
            return self._build_deterministic_sufficiency_result(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason=f"{primary_reason}; used deterministic validator",
            )

    def _build_deterministic_sufficiency_result(
        self,
        *,
        route_family: str,
        risk_level: str,
        required_sources: list[str],
        candidate_supporting_doc_ids: list[str],
        missing_sources: list[str],
        fallback_reason: str | None = None,
    ) -> ContextSufficiencyResult:
        sufficient = not required_sources or not missing_sources
        fallback_route = None
        if not sufficient and route_family == "refund_candidate":
            fallback_route = "request_info"
        elif not sufficient and route_family == "escalation_candidate":
            fallback_route = "status_update"
        reasoning = (
            "Deterministic sufficiency validator found all required evidence."
            if sufficient
            else f"Deterministic sufficiency validator found missing sources: {', '.join(missing_sources)}."
        )
        return ContextSufficiencyResult(
            route_family=route_family,  # type: ignore[arg-type]
            risk_level=risk_level,  # type: ignore[arg-type]
            required_sources=required_sources,
            supporting_doc_ids=candidate_supporting_doc_ids,
            missing_sources=missing_sources,
            sufficient=sufficient,
            fallback_route=fallback_route,  # type: ignore[arg-type]
            reasoning=reasoning,
            decision_source="governance",
            fallback_reason=fallback_reason,
        )

    def assess_sufficiency(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ContextSufficiencyResult:
        route_family = infer_route_family(ticket, triage)
        risk_level = infer_risk_level(ticket, triage)
        requirement_profile = get_requirement_profile(route_family)
        deterministic_missing = _missing_required_sources(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )
        candidate_supporting_doc_ids = _supporting_doc_ids(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
        )

        if self.action_primary_llm_client is None:
            return self._build_deterministic_sufficiency_result(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason="No LLM client configured for context sufficiency; used deterministic validator.",
            )

        try:
            return self._assess_sufficiency_with_llm(
                self.action_primary_llm_client,
                ticket,
                triage,
                customer,
                order,
                policy_hits,
                retrieved_docs,
            )
        except Exception as exc:
            primary_reason = _fallback_reason("上下文充分性判断", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                try:
                    result = self._assess_sufficiency_with_llm(
                        self.action_fallback_llm_client,
                        ticket,
                        triage,
                        customer,
                        order,
                        policy_hits,
                        retrieved_docs,
                    )
                    result.fallback_reason = primary_reason
                    return result
                except Exception as fallback_exc:
                    fallback_reason = (
                        f"{primary_reason}; "
                        f"{_fallback_reason('上下文充分性判断', self.action_fallback_llm_client.source_label, fallback_exc)}; "
                        "used deterministic validator"
                    )
                    return self._build_deterministic_sufficiency_result(
                        route_family=route_family,
                        risk_level=risk_level,
                        required_sources=requirement_profile.required_sources,
                        candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                        missing_sources=deterministic_missing,
                        fallback_reason=fallback_reason,
                    )
            return self._build_deterministic_sufficiency_result(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                missing_sources=deterministic_missing,
                fallback_reason=f"{primary_reason}; used deterministic validator",
            )

    def _build_action_proposal(
        self,
        *,
        action_type: str,
        ticket: TicketRecord,
        triage: TriageResult,
        order: OrderRecord | None,
        rationale: str,
        confidence: float,
        target_status: str | None = None,
        requires_approval: bool | None = None,
        route_family: str = "standard_resolution",
        sufficiency_passed: bool = True,
        supporting_doc_ids: list[str] | None = None,
        missing_sources: list[str] | None = None,
        decision_source: str = "llm_cloud",
        fallback_reason: str | None = None,
    ) -> ActionProposal:
        resolved_target_status = _normalize_target_status(target_status, action_type, triage.category)
        if action_type == "refund":
            tool_name, tool_args = "issue_refund_request", {
                "ticket_id": ticket.ticket_id,
                "order_id": order.order_id if order else ticket.linked_order_id,
                "amount": order.amount if order else None,
                "rationale": "Sufficient refund evidence is available and the request can enter a financial approval flow.",
            }
        elif action_type == "escalation":
            tool_name, tool_args = "create_escalation", {
                "ticket_id": ticket.ticket_id,
                "reason": "Context sufficiency passed for a high-risk escalation candidate.",
                "priority": triage.priority,
            }
        else:
            tool_name, tool_args = "update_ticket_status", {"ticket_id": ticket.ticket_id, "status": resolved_target_status}
        tool_policy = get_tool_approval_policy(tool_name)
        if tool_policy.sensitive:
            resolved_requires_approval = True
        else:
            resolved_requires_approval = bool(requires_approval) if requires_approval is not None else action_type in {"refund", "escalation"}
        return ActionProposal(
            action_type=action_type,  # type: ignore[arg-type]
            target_status=resolved_target_status,
            rationale=rationale,
            requires_approval=resolved_requires_approval,
            suggested_tool=tool_name,
            tool_args=tool_args,
            confidence=round(confidence, 2),
            route_family=route_family,  # type: ignore[arg-type]
            sufficiency_passed=sufficiency_passed,
            supporting_doc_ids=supporting_doc_ids or [],
            missing_sources=missing_sources or [],
            decision_source=decision_source,  # type: ignore[arg-type]
            fallback_reason=fallback_reason,
        )

    def assess_sufficiency(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ContextSufficiencyResult:
        route_family = infer_route_family(ticket, triage)
        risk_level = infer_risk_level(ticket, triage)
        requirement_profile = get_requirement_profile(route_family)
        min_history_strength = requirement_profile.min_history_strength
        deterministic_missing = _missing_required_sources(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
            min_history_strength=min_history_strength,
        )
        candidate_supporting_doc_ids = _supporting_doc_ids(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
            min_history_strength=min_history_strength,
        )
        supporting_history_doc_ids = _supporting_history_doc_ids(retrieved_docs, min_history_strength)
        support_strength = _compute_support_strength(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
            min_history_strength=min_history_strength,
        )
        if self.action_primary_llm_client is None:
            return self._build_deterministic_sufficiency_result_v3(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                supporting_history_doc_ids=supporting_history_doc_ids,
                missing_sources=deterministic_missing,
                support_strength=support_strength,
                fallback_reason="No LLM client configured for context sufficiency; used deterministic validator.",
            )
        try:
            return self._assess_sufficiency_with_llm(
                self.action_primary_llm_client,
                ticket,
                triage,
                customer,
                order,
                policy_hits,
                retrieved_docs,
            )
        except Exception as exc:
            primary_reason = _fallback_reason("上下文充分性判断", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                try:
                    result = self._assess_sufficiency_with_llm(
                        self.action_fallback_llm_client,
                        ticket,
                        triage,
                        customer,
                        order,
                        policy_hits,
                        retrieved_docs,
                    )
                    result.fallback_reason = primary_reason
                    return result
                except Exception as fallback_exc:
                    fallback_reason = (
                        f"{primary_reason}; "
                        f"{_fallback_reason('上下文充分性判断', self.action_fallback_llm_client.source_label, fallback_exc)}; "
                        "used deterministic validator"
                    )
                    return self._build_deterministic_sufficiency_result_v3(
                        route_family=route_family,
                        risk_level=risk_level,
                        required_sources=requirement_profile.required_sources,
                        candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                        supporting_history_doc_ids=supporting_history_doc_ids,
                        missing_sources=deterministic_missing,
                        support_strength=support_strength,
                        fallback_reason=fallback_reason,
                    )
            return self._build_deterministic_sufficiency_result_v3(
                route_family=route_family,
                risk_level=risk_level,
                required_sources=requirement_profile.required_sources,
                candidate_supporting_doc_ids=candidate_supporting_doc_ids,
                supporting_history_doc_ids=supporting_history_doc_ids,
                missing_sources=deterministic_missing,
                support_strength=support_strength,
                fallback_reason=f"{primary_reason}; used deterministic validator",
            )

    def _assess_sufficiency_with_llm(
        self,
        llm_client: OpenAICompatClient,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
    ) -> ContextSufficiencyResult:
        route_family = infer_route_family(ticket, triage)
        risk_level = infer_risk_level(ticket, triage)
        requirement_profile = get_requirement_profile(route_family)
        min_history_strength = requirement_profile.min_history_strength
        deterministic_missing = _missing_required_sources(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
            min_history_strength=min_history_strength,
        )
        candidate_supporting_doc_ids = _supporting_doc_ids(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
            min_history_strength=min_history_strength,
        )
        supporting_history_doc_ids = _supporting_history_doc_ids(retrieved_docs, min_history_strength)
        support_strength = _compute_support_strength(
            required_sources=requirement_profile.required_sources,
            order=order,
            policy_hits=policy_hits,
            retrieved_docs=retrieved_docs,
            min_history_strength=min_history_strength,
        )
        doc_id_inventory = {*(f"{doc.source_type}:{doc.doc_id}" for doc in retrieved_docs), *(f"policy:{doc.doc_id}" for doc in policy_hits)}
        if order is not None and order.order_id:
            doc_id_inventory.add(f"order:{order.order_id}")
        llm_prompt = (
            "Return JSON only with keys sufficient, supporting_doc_ids, missing_sources, reasoning. "
            "supporting_doc_ids must only use ids that appear in the provided evidence inventory. "
            "missing_sources must only use values from required_sources. "
            "deterministic_missing_sources is authoritative for whether a required source type is present; "
            "do not mark a source missing if it is not listed there. "
            "For escalation candidates, weak history evidence below the configured threshold is not sufficient.\n"
            f"ticket={ticket.model_dump(mode='json')}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"customer={customer.model_dump(mode='json') if customer else 'null'}\n"
            f"order={order.model_dump(mode='json') if order else 'null'}\n"
            f"route_family={route_family}\n"
            f"risk_level={risk_level}\n"
            f"required_sources={json.dumps(requirement_profile.required_sources, ensure_ascii=False)}\n"
            f"optional_sources={json.dumps(requirement_profile.optional_sources, ensure_ascii=False)}\n"
            f"min_history_strength={min_history_strength}\n"
            f"current_support_strength={support_strength}\n"
            f"deterministic_missing_sources={json.dumps(deterministic_missing, ensure_ascii=False)}\n"
            f"evidence_inventory={json.dumps(sorted(doc_id_inventory), ensure_ascii=False)}\n"
            f"candidate_supporting_history_doc_ids={json.dumps(supporting_history_doc_ids, ensure_ascii=False)}\n"
            f"policy_evidence={[doc.snippet for doc in policy_hits[:3]]}\n"
            f"context_evidence={[f'{doc.source_type}:{doc.doc_id}' for doc in retrieved_docs[:8]]}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are a context-sufficiency judge for enterprise ticket workflows.",
            user_prompt=llm_prompt,
            validator=lambda value: value is not None
            and {"sufficient", "supporting_doc_ids", "missing_sources", "reasoning"}.issubset(value),
            schema_hint="Required keys: sufficient, supporting_doc_ids, missing_sources, reasoning.",
        )
        deterministic_missing_set = set(deterministic_missing)
        llm_missing = [
            source
            for source in _extract_unsupported_claims(payload.get("missing_sources"))
            if source in deterministic_missing_set
        ]
        llm_supporting = [doc_id for doc_id in _extract_unsupported_claims(payload.get("supporting_doc_ids")) if doc_id in doc_id_inventory]
        missing_sources = sorted({*deterministic_missing, *llm_missing})
        supporting_doc_ids: list[str] = []
        for doc_id in [*candidate_supporting_doc_ids, *llm_supporting]:
            if doc_id not in supporting_doc_ids:
                supporting_doc_ids.append(doc_id)
        supporting_history_doc_ids = [doc_id for doc_id in supporting_history_doc_ids if doc_id in supporting_doc_ids]
        sufficient = True if not requirement_profile.required_sources else not missing_sources
        if deterministic_missing and requirement_profile.required_sources and not _normalize_bool(payload.get("sufficient", sufficient)):
            sufficient = False
        if not requirement_profile.required_sources:
            sufficient = True
        fallback_route = None
        if not sufficient and route_family == "refund_candidate":
            fallback_route = "request_info"
        elif not sufficient and route_family == "escalation_candidate":
            fallback_route = "status_update"
        return ContextSufficiencyResult(
            route_family=route_family,
            risk_level=risk_level,  # type: ignore[arg-type]
            required_sources=requirement_profile.required_sources,
            supporting_doc_ids=supporting_doc_ids,
            supporting_history_doc_ids=supporting_history_doc_ids,
            missing_sources=missing_sources,
            sufficient=sufficient,
            fallback_route=fallback_route,  # type: ignore[arg-type]
            support_strength=support_strength,
            reasoning=str(payload.get("reasoning") or "Context sufficiency assessed."),
            decision_source=llm_client.source_label,  # type: ignore[arg-type]
        )

    def _repair_action_to_allowed(
        self,
        llm_client: OpenAICompatClient,
        *,
        prompt_context: str,
        allowed_actions: list[str],
    ) -> dict[str, object]:
        repair_prompt = (
            "Return JSON only with keys action_type, target_status, requires_approval, rationale, confidence. "
            f"Allowed action_type must be one of {allowed_actions}. "
            f"Original context:\n{prompt_context}"
        )
        return _call_structured_json(
            llm_client,
            system_prompt="You repair invalid action decisions for an enterprise ticket workflow.",
            user_prompt=repair_prompt,
            validator=_action_payload_valid,
            schema_hint=(
                "Required keys: action_type, target_status, requires_approval, rationale, confidence. "
                f"Allowed action_type must be one of {allowed_actions}."
            ),
        )

    def _build_deterministic_action_fallback(
        self,
        *,
        ticket: TicketRecord,
        triage: TriageResult,
        order: OrderRecord | None,
        sufficiency_result: ContextSufficiencyResult,
        action_route: ActionRoutingDecision,
        fallback_reason: str,
    ) -> ActionProposal:
        if action_route.forced_action_type is not None:
            action_type = action_route.forced_action_type
            target_status = action_route.forced_target_status
            confidence = 1.0
            rationale = f"{action_route.reasoning} Deterministic constrained fallback was used because the structured action model failed."
        elif sufficiency_result.route_family == "refund_candidate":
            action_type = "refund"
            target_status = "pending_finance"
            confidence = 0.78
            rationale = "Refund route had sufficient evidence, so the workflow fell back to a deterministic refund action after the structured action model failed."
        elif sufficiency_result.route_family == "escalation_candidate":
            action_type = "escalation"
            target_status = "escalated"
            confidence = 0.78
            rationale = "Escalation route had sufficient evidence, so the workflow fell back to a deterministic escalation action after the structured action model failed."
        elif triage.category in {"technical_issue", "account_access"} and "troubleshoot" in action_route.allowed_actions:
            action_type = "troubleshoot"
            target_status = "investigating"
            confidence = 0.66
            rationale = "The workflow fell back to deterministic troubleshooting because the structured action model failed under a standard-resolution route."
        elif "status_update" in action_route.allowed_actions:
            action_type = "status_update"
            target_status = "monitoring" if triage.category == "delivery_issue" else "in_progress"
            confidence = 0.62
            rationale = "The workflow fell back to a deterministic status update because the structured action model failed under a standard-resolution route."
        else:
            action_type = "request_info"
            target_status = "waiting_on_customer"
            confidence = 0.6
            rationale = "The workflow fell back to requesting more information because the structured action model failed and no safer constrained action was available."

        return self._build_action_proposal(
            action_type=action_type,
            ticket=ticket,
            triage=triage,
            order=order,
            rationale=rationale,
            confidence=confidence,
            target_status=target_status,
            requires_approval=None,
            route_family=sufficiency_result.route_family,
            sufficiency_passed=sufficiency_result.sufficient,
            supporting_doc_ids=sufficiency_result.supporting_doc_ids,
            missing_sources=sufficiency_result.missing_sources,
            decision_source="governance",
            fallback_reason=fallback_reason,
        )

    def propose_action(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
        sufficiency_result: ContextSufficiencyResult,
        action_route: ActionRoutingDecision,
    ) -> ActionProposal:
        if action_route.forced_action_type is not None:
            return self._build_action_proposal(
                action_type=action_route.forced_action_type,
                ticket=ticket,
                triage=triage,
                order=order,
                rationale=action_route.reasoning,
                confidence=1.0,
                target_status=action_route.forced_target_status,
                requires_approval=False,
                route_family=sufficiency_result.route_family,
                sufficiency_passed=sufficiency_result.sufficient,
                supporting_doc_ids=sufficiency_result.supporting_doc_ids,
                missing_sources=sufficiency_result.missing_sources,
                decision_source="governance",
            )

        if self.action_primary_llm_client is None:
            return self._build_deterministic_action_fallback(
                ticket=ticket,
                triage=triage,
                order=order,
                sufficiency_result=sufficiency_result,
                action_route=action_route,
                fallback_reason="No LLM client configured for action proposal; used deterministic constrained fallback",
            )
        try:
            return self._propose_action_with_llm(
                self.action_primary_llm_client,
                ticket,
                triage,
                customer,
                order,
                policy_hits,
                retrieved_docs,
                sufficiency_result,
                action_route,
            )
        except Exception as exc:
            primary_reason = _fallback_reason("动作建议", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                try:
                    result = self._propose_action_with_llm(
                        self.action_fallback_llm_client,
                        ticket,
                        triage,
                        customer,
                        order,
                        policy_hits,
                        retrieved_docs,
                        sufficiency_result,
                        action_route,
                    )
                    result.fallback_reason = primary_reason
                    return result
                except Exception as fallback_exc:
                    fallback_reason = (
                        f"{primary_reason}; "
                        f"{_fallback_reason('动作建议', self.action_fallback_llm_client.source_label, fallback_exc)}; "
                        "used deterministic constrained fallback"
                    )
                    return self._build_deterministic_action_fallback(
                        ticket=ticket,
                        triage=triage,
                        order=order,
                        sufficiency_result=sufficiency_result,
                        action_route=action_route,
                        fallback_reason=fallback_reason,
                    )
            return self._build_deterministic_action_fallback(
                ticket=ticket,
                triage=triage,
                order=order,
                sufficiency_result=sufficiency_result,
                action_route=action_route,
                fallback_reason=f"{primary_reason}; used deterministic constrained fallback",
            )

    def _propose_action_with_llm(
        self,
        llm_client: OpenAICompatClient,
        ticket: TicketRecord,
        triage: TriageResult,
        customer: CustomerProfile | None,
        order: OrderRecord | None,
        policy_hits: list[RetrievedDoc],
        retrieved_docs: list[RetrievedDoc],
        sufficiency_result: ContextSufficiencyResult,
        action_route: ActionRoutingDecision,
    ) -> ActionProposal:
        allowed_actions = [str(action) for action in action_route.allowed_actions]
        evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:6])
        prompt_context = (
            f"channel={ticket.channel}\ncustomer_tier={ticket.customer_tier}\nproduct={ticket.product}\n"
            f"linked_order_id={ticket.linked_order_id or 'none'}\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"customer={customer.model_dump(mode='json') if customer else 'null'}\n"
            f"order={order.model_dump(mode='json') if order else 'null'}\n"
            f"route_family={sufficiency_result.route_family}\n"
            f"sufficiency={sufficiency_result.model_dump(mode='json')}\n"
            f"allowed_actions={allowed_actions}\n"
            f"policies={[doc.snippet for doc in policy_hits[:5]]}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        if llm_client.source_label == "llm_minimind":
            prompt = (
                "请基于工单、分诊结果和上下文充分性结果，给出下一步处理动作，并只输出 JSON。"
                "必须包含 action_type、approval_required、target_status。"
                f"action_type 只能从 {allowed_actions} 中选择。\n"
                f"{prompt_context}"
            )
            payload = _call_structured_json(
                llm_client,
                system_prompt="You are MiniMind-Ticket structured adapter. Only solve structured enterprise ticket tasks and always return strict JSON.",
                user_prompt=prompt,
                validator=_local_action_payload_valid,
                schema_hint=(
                    "Required keys: action_type, approval_required, target_status. "
                    f"Allowed action_type must be one of {allowed_actions}. "
                    "Allowed target_status: in_progress, investigating, monitoring, waiting_on_customer, pending_finance, pending_human, escalated."
                ),
            )
            action_type = _normalize_action_type(payload.get("action_type"))
            if action_type not in allowed_actions:
                payload = self._repair_action_to_allowed(llm_client, prompt_context=prompt_context, allowed_actions=allowed_actions)
                action_type = _normalize_action_type(payload.get("action_type"))
            rationale = "MiniMind 结构化适配器在 sufficiency-first 约束下生成动作建议。"
            target_status = _normalize_target_status(payload.get("target_status"), action_type, triage.category)
            action_type, target_status, rationale, governance_repaired = _repair_unnecessary_standard_request_info(
                action_type=action_type,
                target_status=target_status,
                rationale=rationale,
                triage=triage,
                sufficiency_result=sufficiency_result,
            )
            return self._build_action_proposal(
                action_type=action_type,
                ticket=ticket,
                triage=triage,
                order=order,
                rationale=rationale,
                confidence=0.72,
                target_status=target_status,
                requires_approval=_normalize_bool(payload.get("approval_required")),
                route_family=sufficiency_result.route_family,
                sufficiency_passed=sufficiency_result.sufficient,
                supporting_doc_ids=sufficiency_result.supporting_doc_ids,
                missing_sources=sufficiency_result.missing_sources,
                decision_source="governance" if governance_repaired else llm_client.source_label,
            )

        prompt = (
            "Return JSON only with keys action_type, target_status, requires_approval, rationale, confidence. "
            f"Allowed action_type must be one of {allowed_actions}. "
            "Allowed target_status: in_progress, investigating, monitoring, waiting_on_customer, pending_finance, pending_human, escalated. "
            "Use the sufficiency result as a hard decision boundary for high-risk actions.\n"
            f"{prompt_context}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are an enterprise ticket action-decision model operating under sufficiency-first governance.",
            user_prompt=prompt,
            validator=_action_payload_valid,
            schema_hint=(
                "Required keys: action_type, target_status, requires_approval, rationale, confidence. "
                f"Allowed action_type must be one of {allowed_actions}."
            ),
        )
        action_type = _normalize_action_type(payload.get("action_type"))
        if action_type not in allowed_actions:
            payload = self._repair_action_to_allowed(llm_client, prompt_context=prompt_context, allowed_actions=allowed_actions)
            action_type = _normalize_action_type(payload.get("action_type"))
        rationale = str(payload.get("rationale") or "LLM generated a constrained action proposal.")
        target_status = _normalize_target_status(payload.get("target_status"), action_type, triage.category)
        action_type, target_status, rationale, governance_repaired = _repair_unnecessary_standard_request_info(
            action_type=action_type,
            target_status=target_status,
            rationale=rationale,
            triage=triage,
            sufficiency_result=sufficiency_result,
        )
        return self._build_action_proposal(
            action_type=action_type,
            ticket=ticket,
            triage=triage,
            order=order,
            rationale=rationale,
            confidence=_normalize_confidence(payload.get("confidence"), 0.68),
            target_status=target_status,
            requires_approval=_normalize_bool(payload.get("requires_approval")),
            route_family=sufficiency_result.route_family,
            sufficiency_passed=sufficiency_result.sufficient,
            supporting_doc_ids=sufficiency_result.supporting_doc_ids,
            missing_sources=sufficiency_result.missing_sources,
            decision_source="governance" if governance_repaired else llm_client.source_label,
        )

    def draft_reply(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        proposed_action: ActionProposal,
        execution_result: object,
        retrieved_docs: list[RetrievedDoc],
        templates: list[str],
        review_decision: ReviewDecision | None = None,
    ) -> TicketResponse:
        del templates
        if _should_use_governance_safe_reply(proposed_action):
            response = TicketResponse(
                customer_reply=_build_conservative_local_reply(ticket, triage, proposed_action, execution_result),
                internal_note=(
                    proposed_action.rationale
                    or "Governance fallback route generated a deterministic safe reply because critical evidence was insufficient."
                ),
                status=proposed_action.target_status,
                citations=_collect_top_citations(retrieved_docs),
                draft_source="governance",
                fallback_reason="governance_safe_fallback",
            )
            response.review_note = "Reply generation short-circuited to a governance-safe template because the workflow had already routed the ticket conservatively."
            return response
        if self.draft_primary_llm_client is None:
            return TicketResponse(
                customer_reply=_build_conservative_local_reply(ticket, triage, proposed_action, execution_result),
                internal_note=(
                    "No LLM client configured for reply drafting; generated a deterministic conservative reply "
                    "from the approved action route."
                ),
                status=proposed_action.target_status,
                citations=_collect_top_citations(retrieved_docs),
                draft_source="governance",
                fallback_reason="No LLM client configured for reply drafting; used deterministic conservative reply",
            )
        try:
            return self._draft_with_llm(
                self.draft_primary_llm_client,
                ticket,
                triage,
                proposed_action,
                execution_result,
                retrieved_docs,
                review_decision,
            )
        except Exception as exc:
            primary_reason = _fallback_reason("回复生成", self.draft_primary_llm_client.source_label, exc)
            if self.draft_fallback_llm_client is not None:
                result = self._draft_with_llm(
                    self.draft_fallback_llm_client,
                    ticket,
                    triage,
                    proposed_action,
                    execution_result,
                    retrieved_docs,
                    review_decision,
                )
                result.fallback_reason = primary_reason
                return result
            raise RuntimeError(primary_reason) from exc

    def fact_check_reply(
        self,
        ticket: TicketRecord,
        triage: TriageResult,
        proposed_action: ActionProposal,
        execution_result: object,
        retrieved_docs: list[RetrievedDoc],
        response: TicketResponse,
    ) -> tuple[TicketResponse, ReplyFactCheckResult]:
        if response.draft_source == "governance" and _should_use_governance_safe_reply(proposed_action):
            fact_check = ReplyFactCheckResult(
                grounded=True,
                policy_safe=True,
                hallucination_risk=False,
                supported_claim_ratio=1.0,
                unsupported_claims=[],
                rewrite_needed=False,
                review_note="Governance-safe fallback reply bypassed LLM fact-check because it was generated from a deterministic conservative route.",
                passed=True,
                rewritten=False,
                downgraded=False,
            )
            response.review_passed = True
            response.review_grounded = True
            response.review_policy_safe = True
            response.review_hallucination_risk = False
            response.review_supported_claim_ratio = 1.0
            response.review_note = fact_check.review_note
            return response, fact_check
        if not self.reply_review_enabled:
            result = ReplyFactCheckResult(
                grounded=True,
                policy_safe=True,
                hallucination_risk=False,
                supported_claim_ratio=1.0,
                unsupported_claims=[],
                rewrite_needed=False,
                review_note="Reply review is disabled.",
                passed=True,
            )
            response.review_passed = True
            response.review_grounded = True
            response.review_policy_safe = True
            response.review_hallucination_risk = False
            response.review_supported_claim_ratio = 1.0
            response.review_note = result.review_note
            return response, result

        claim_inventory = _build_reply_claim_inventory(
            ticket,
            triage,
            proposed_action,
            execution_result,
            retrieved_docs,
        )
        deterministic_review = _deterministic_reply_fact_check(
            response.customer_reply,
            claim_inventory,
            threshold=self.reply_supported_claim_threshold,
        )
        review_payload: dict[str, object] | None = None
        checked_response = response
        fact_check: ReplyFactCheckResult | None = None
        was_rewritten = False
        was_downgraded = False
        if bool(deterministic_review.get("decisive")):
            review_payload = deterministic_review
            fact_check = self._build_fact_check_result(review_payload)
            response.review_passed = fact_check.passed
            response.review_grounded = fact_check.grounded
            response.review_policy_safe = fact_check.policy_safe
            response.review_hallucination_risk = fact_check.hallucination_risk
            response.review_supported_claim_ratio = fact_check.supported_claim_ratio
            response.review_note = fact_check.review_note
            if fact_check.passed:
                return response, fact_check

        verifier_client = self.draft_fallback_llm_client or self.draft_primary_llm_client
        if fact_check is None:
            if verifier_client is None:
                fact_check = ReplyFactCheckResult(
                    grounded=False,
                    policy_safe=True,
                    hallucination_risk=True,
                    supported_claim_ratio=0.0,
                    unsupported_claims=["reply_fact_check_unavailable"],
                    rewrite_needed=True,
                    review_note="No verifier LLM client configured; downgraded conservatively.",
                    passed=False,
                    rewritten=False,
                    downgraded=True,
                )
                checked_response = TicketResponse(
                    customer_reply=_build_conservative_local_reply(ticket, triage, proposed_action, execution_result),
                    internal_note=response.internal_note,
                    status=response.status,
                    citations=response.citations,
                    draft_source=response.draft_source,
                    fallback_reason=response.fallback_reason,
                )
                was_downgraded = True
            else:
                try:
                    review_payload = self._review_reply_with_llm(
                        verifier_client,
                        ticket,
                        triage,
                        proposed_action,
                        execution_result,
                        retrieved_docs,
                        response,
                    )
                    fact_check = self._build_fact_check_result(review_payload)
                except Exception as exc:
                    review_payload = {
                        "grounded": False,
                        "policy_safe": True,
                        "hallucination_risk": True,
                        "supported_claim_ratio": 0.0,
                        "unsupported_claims": ["reply_fact_check_failed"],
                        "review_note": f"Fact-check verifier failed: {type(exc).__name__}",
                        "review_passed": False,
                        "needs_rewrite": True,
                    }
                    fact_check = self._build_fact_check_result(review_payload)

        if fact_check.rewrite_needed and not fact_check.passed:
            repair_client = self._repair_client_for_reply(response.draft_source)
            if repair_client is not None and response.draft_source == "llm_cloud":
                try:
                    checked_response = self._repair_reply_with_llm(
                        repair_client,
                        ticket,
                        triage,
                        proposed_action,
                        execution_result,
                        retrieved_docs,
                        response,
                        review_payload,
                    )
                    post_repair_payload = self._review_reply_with_llm(
                        verifier_client,
                        ticket,
                        triage,
                        proposed_action,
                        execution_result,
                        retrieved_docs,
                        checked_response,
                    )
                    fact_check = self._build_fact_check_result(post_repair_payload, rewritten=True)
                    was_rewritten = True
                except Exception as exc:
                    fact_check = ReplyFactCheckResult(
                        grounded=False,
                        policy_safe=fact_check.policy_safe,
                        hallucination_risk=True,
                        supported_claim_ratio=0.0,
                        unsupported_claims=["reply_repair_failed"],
                        rewrite_needed=True,
                        review_note=f"{fact_check.review_note} Repair path failed: {type(exc).__name__}.",
                        passed=False,
                        rewritten=False,
                        downgraded=True,
                    )

            if not fact_check.passed:
                checked_response = TicketResponse(
                    customer_reply=_build_conservative_local_reply(ticket, triage, proposed_action, execution_result),
                    internal_note=response.internal_note,
                    status=response.status,
                    citations=response.citations,
                    draft_source=response.draft_source,
                    fallback_reason=response.fallback_reason,
                )
                was_downgraded = True
                fact_check = ReplyFactCheckResult(
                    grounded=False,
                    policy_safe=fact_check.policy_safe,
                    hallucination_risk=fact_check.hallucination_risk,
                    supported_claim_ratio=fact_check.supported_claim_ratio,
                    unsupported_claims=fact_check.unsupported_claims,
                    rewrite_needed=True,
                    review_note=f"{fact_check.review_note} Downgraded to conservative reply after fact-check failure.",
                    passed=False,
                    rewritten=fact_check.rewritten,
                    downgraded=True,
                )

        final_review = _revalidate_final_reply_against_contract(
            ticket,
            triage,
            proposed_action,
            execution_result,
            retrieved_docs,
            checked_response,
            threshold=self.reply_supported_claim_threshold,
        )
        if bool(final_review.get("decisive")):
            fact_check = self._build_fact_check_result(final_review, rewritten=was_rewritten or fact_check.rewritten)
            fact_check.downgraded = was_downgraded or fact_check.downgraded

        checked_response.review_passed = fact_check.passed
        checked_response.review_grounded = fact_check.grounded
        checked_response.review_policy_safe = fact_check.policy_safe
        checked_response.review_hallucination_risk = fact_check.hallucination_risk
        checked_response.review_supported_claim_ratio = fact_check.supported_claim_ratio
        checked_response.review_note = fact_check.review_note
        return checked_response, fact_check

    def _repair_client_for_reply(self, draft_source: str) -> OpenAICompatClient | None:
        clients = [self.draft_primary_llm_client, self.draft_fallback_llm_client]
        for client in clients:
            if client is not None and client.source_label == draft_source:
                return client
        return self.draft_primary_llm_client or self.draft_fallback_llm_client

    @staticmethod
    def _build_fact_check_result(payload: dict[str, object], rewritten: bool = False) -> ReplyFactCheckResult:
        return ReplyFactCheckResult(
            grounded=bool(payload["grounded"]),
            policy_safe=bool(payload["policy_safe"]),
            hallucination_risk=bool(payload["hallucination_risk"]),
            supported_claim_ratio=float(payload["supported_claim_ratio"]),
            unsupported_claims=list(payload["unsupported_claims"]),
            rewrite_needed=bool(payload["needs_rewrite"]),
            review_note=str(payload["review_note"]),
            passed=bool(payload["review_passed"]),
            rewritten=rewritten,
            downgraded=False,
        )

    def _review_reply_with_llm(
        self,
        llm_client: OpenAICompatClient,
        ticket: TicketRecord,
        triage: TriageResult,
        proposed_action: ActionProposal,
        execution_result: object,
        retrieved_docs: list[RetrievedDoc],
        response: TicketResponse,
    ) -> dict[str, object]:
        claim_inventory = _build_reply_claim_inventory(
            ticket,
            triage,
            proposed_action,
            execution_result,
            retrieved_docs,
        )
        evidence = "\n".join(f"- {line}" for line in claim_inventory["evidence_summary"][:6])
        prompt = (
            "Return JSON only with keys grounded, policy_safe, hallucination_risk, supported_claim_ratio, unsupported_claims, review_note, needs_rewrite. "
            "supported_claim_ratio must be between 0 and 1. unsupported_claims must be a list of short strings. "
            "Evaluate the reply claim by claim. Judge whether each claim stays within supported_claims, covers route-critical required_claims, follows status_contract, and avoids forbidden_claims.\n"
            f"reply={response.customer_reply}\n"
            f"contract_summary={claim_inventory['contract_summary']}\n"
            f"status_contract={json.dumps(claim_inventory['status_contract'], ensure_ascii=False)}\n"
            f"supported_claims={json.dumps(claim_inventory['supported_claims'], ensure_ascii=False)}\n"
            f"required_claims={json.dumps(claim_inventory['required_claims'], ensure_ascii=False)}\n"
            f"forbidden_claims={json.dumps(claim_inventory['forbidden_claims'], ensure_ascii=False)}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are a grounded-reply verifier in an enterprise ticket workflow.",
            user_prompt=prompt,
            validator=lambda value: value is not None
            and {"grounded", "policy_safe", "hallucination_risk", "supported_claim_ratio", "unsupported_claims", "review_note", "needs_rewrite"}.issubset(value),
            schema_hint="Required keys: grounded, policy_safe, hallucination_risk, supported_claim_ratio, unsupported_claims, review_note, needs_rewrite.",
        )
        grounded = _normalize_bool(payload.get("grounded"))
        policy_safe = _normalize_bool(payload.get("policy_safe"))
        hallucination_risk = _normalize_bool(payload.get("hallucination_risk"))
        supported_claim_ratio = _normalize_confidence(payload.get("supported_claim_ratio"), 0.0)
        needs_rewrite = _normalize_bool(payload.get("needs_rewrite")) or supported_claim_ratio < self.reply_supported_claim_threshold
        unsupported_claims = _extract_unsupported_claims(payload.get("unsupported_claims"))
        return {
            "grounded": grounded,
            "policy_safe": policy_safe,
            "hallucination_risk": hallucination_risk,
            "supported_claim_ratio": supported_claim_ratio,
            "unsupported_claims": unsupported_claims,
            "review_note": str(payload.get("review_note") or "LLM completed reply verification."),
            "review_passed": grounded and policy_safe and not hallucination_risk and supported_claim_ratio >= self.reply_supported_claim_threshold,
            "needs_rewrite": needs_rewrite,
        }

    def _repair_reply_with_llm(
        self,
        llm_client: OpenAICompatClient,
        ticket: TicketRecord,
        triage: TriageResult,
        proposed_action: ActionProposal,
        execution_result: object,
        retrieved_docs: list[RetrievedDoc],
        response: TicketResponse,
        review_payload: dict[str, object],
    ) -> TicketResponse:
        claim_inventory = _build_reply_claim_inventory(
            ticket,
            triage,
            proposed_action,
            execution_result,
            retrieved_docs,
        )
        evidence = "\n".join(f"- {line}" for line in claim_inventory["evidence_summary"][:6])
        prompt = (
            "Return JSON only with keys customer_reply, internal_note, status. "
            "Rewrite the reply to remove unsupported claims and keep it conservative and evidence-grounded. "
            "Preserve only claims allowed by supported_claims and status_contract. "
            "If unsure, move closer to safe_template instead of inventing detail.\n"
            f"original_reply={response.customer_reply}\n"
            f"contract_summary={claim_inventory['contract_summary']}\n"
            f"status_contract={json.dumps(claim_inventory['status_contract'], ensure_ascii=False)}\n"
            f"supported_claims={json.dumps(claim_inventory['supported_claims'], ensure_ascii=False)}\n"
            f"required_claims={json.dumps(claim_inventory['required_claims'], ensure_ascii=False)}\n"
            f"forbidden_claims={json.dumps(claim_inventory['forbidden_claims'], ensure_ascii=False)}\n"
            f"unsupported_claims={json.dumps(review_payload.get('unsupported_claims', []), ensure_ascii=False)}\n"
            f"review_note={review_payload.get('review_note')}\n"
            f"safe_template={claim_inventory['template_reply']}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        payload = _call_structured_json(
            llm_client,
            system_prompt="You are a reply-repair model in an enterprise ticket workflow.",
            user_prompt=prompt,
            validator=_reply_payload_valid,
            schema_hint="Required keys: customer_reply, internal_note, status.",
        )
        return TicketResponse(
            customer_reply=str(payload.get("customer_reply") or response.customer_reply),
            internal_note=str(payload.get("internal_note") or response.internal_note),
            status=_normalize_ticket_status(str(payload.get("status") or response.status)),
            citations=response.citations,
            draft_source=response.draft_source,
            fallback_reason=response.fallback_reason,
        )
