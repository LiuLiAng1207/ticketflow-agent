from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable

from .llm import OpenAICompatClient
from .models import (
    ActionProposal,
    Citation,
    CustomerProfile,
    OrderRecord,
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


def _collect_top_citations(retrieved_docs: list[RetrievedDoc]) -> list[Citation]:
    return [citation for doc in retrieved_docs[:3] for citation in doc.citations[:1]]


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
    sla_risk_review_enabled: bool = True
    sla_risk_threshold: float = 0.72

    def analyze(self, ticket: TicketRecord) -> TriageResult:
        if self.primary_llm_client is None:
            raise RuntimeError("No LLM client configured for triage")
        try:
            return self._analyze_with_llm(ticket, self.primary_llm_client)
        except Exception as exc:
            primary_reason = _fallback_reason("分诊", self.primary_llm_client.source_label, exc)
            if self.fallback_llm_client is not None:
                result = self._analyze_with_llm(ticket, self.fallback_llm_client)
                result.fallback_reason = primary_reason
                return result
            raise RuntimeError(primary_reason) from exc

    def _analyze_with_llm(self, ticket: TicketRecord, llm_client: OpenAICompatClient) -> TriageResult:
        if llm_client.source_label == "llm_minimind":
            prompt = (
                "请对下面工单做结构化分诊，并只输出 JSON。必须包含 category、priority、urgency、sla_risk。\n"
                f"channel={ticket.channel}\n"
                f"customer_tier={ticket.customer_tier}\n"
                f"product={ticket.product}\n"
                f"linked_order_id={ticket.linked_order_id or 'none'}\n"
                f"title_cn={ticket.title}\n"
                f"body_cn={ticket.body}"
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
            raise RuntimeError("No LLM client configured for action proposal")
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
        if llm_client.source_label == "llm_minimind":
            prompt = (
                "请基于工单和分诊结果给出下一步处理动作，并只输出 JSON。必须包含 action_type、approval_required、target_status。\n"
                f"channel={ticket.channel}\n"
                f"customer_tier={ticket.customer_tier}\n"
                f"product={ticket.product}\n"
                f"linked_order_id={ticket.linked_order_id or 'none'}\n"
                f"title_cn={ticket.title}\n"
                f"body_cn={ticket.body}\n"
                f"triage={json.dumps({'category': triage.category, 'priority': triage.priority, 'sla_risk': triage.sla_risk}, ensure_ascii=False)}"
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
            return self._build_action_proposal(
                action_type=action_type,
                ticket=ticket,
                triage=triage,
                order=order,
                rationale="MiniMind 结构化适配器生成动作建议。",
                confidence=0.72,
                target_status=_normalize_target_status(payload.get("target_status"), action_type, triage.category),
                requires_approval=_normalize_bool(payload.get("approval_required")),
                decision_source=llm_client.source_label,
            )

        evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:6])
        prompt = (
            "Return JSON only with keys action_type, target_status, requires_approval, rationale, confidence. "
            "Allowed action_type: refund, escalation, request_info, status_update, troubleshoot. "
            "Allowed target_status: in_progress, investigating, monitoring, waiting_on_customer, pending_finance, pending_human, escalated. "
            "If refund evidence is incomplete or no valid order exists, prefer request_info instead of refund. "
            "For normal account_access issues, prefer troubleshoot instead of escalation. "
            f"channel={ticket.channel}\ncustomer_tier={ticket.customer_tier}\nproduct={ticket.product}\n"
            f"linked_order_id={ticket.linked_order_id or 'none'}\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"customer={customer.model_dump(mode='json') if customer else 'null'}\n"
            f"order={order.model_dump(mode='json') if order else 'null'}\n"
            f"policies={[doc.snippet for doc in policy_hits[:5]]}\n"
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
        return self._build_action_proposal(
            action_type=action_type,
            ticket=ticket,
            triage=triage,
            order=order,
            rationale=str(payload.get("rationale") or "LLM generated the next action proposal."),
            confidence=_normalize_confidence(payload.get("confidence"), 0.68),
            target_status=_normalize_target_status(payload.get("target_status"), action_type, triage.category),
            requires_approval=_normalize_bool(payload.get("requires_approval")),
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
        if llm_client.source_label == "llm_minimind":
            execution_payload = getattr(execution_result, "__dict__", execution_result)
            evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:4])
            prompt = (
                "下面是一张企业工单的处理信息，请生成一条发给客户的中文客服回复。"
                "要求礼貌、简洁、保守，不要输出 JSON，不要承诺尚未执行的退款、修复或发货结果。\n"
                f"title_cn={ticket.title}\n"
                f"body_cn={ticket.body}\n"
                f"action_type={proposed_action.action_type}\n"
                f"target_status={proposed_action.target_status}\n"
                f"execution={execution_payload}\n"
                f"evidence=\n{evidence or '- none'}"
            )
            reply_text = llm_client.chat_text(
                "你是中文客服回复模型。请基于真实工单事实生成下一句客服回复，保持礼貌、简洁、自然，不编造新事实。",
                prompt,
            ).strip()
            reply_text = re.sub(r"^\s*```[a-zA-Z]*", "", reply_text).strip()
            reply_text = re.sub(r"```\s*$", "", reply_text).strip()
            if "\n" in reply_text:
                reply_text = next((line.strip() for line in reply_text.splitlines() if line.strip()), "")
            return TicketResponse(
                customer_reply=reply_text or "我们已经收到您的问题，会尽快为您核实处理。",
                internal_note=proposed_action.rationale or "本地回复模型已生成客户沟通文案。",
                status=proposed_action.target_status,
                citations=_collect_top_citations(retrieved_docs),
                draft_source=llm_client.source_label,  # type: ignore[arg-type]
                fallback_reason=None,
            )

        execution_payload = getattr(execution_result, "__dict__", execution_result)
        evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:6])
        prompt = (
            "Return JSON only with keys customer_reply, internal_note, status. "
            "Write customer_reply and internal_note in Chinese. "
            "Never claim a refund, fix, or delivery completion unless execution.status is executed.\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"action={proposed_action.model_dump(mode='json')}\n"
            f"execution={execution_payload}\n"
            f"review={review_decision.model_dump(mode='json') if review_decision else 'null'}\n"
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
