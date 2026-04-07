from __future__ import annotations

import json
import re
from dataclasses import dataclass

from .llm import OpenAICompatClient
from .models import ActionProposal, Citation, CustomerProfile, OrderRecord, RetrievedDoc, ReviewDecision, TicketRecord, TicketResponse, TriageResult

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


def _contains_any(text: str, keywords: list[str]) -> int:
    lowered = text.lower()
    return sum(1 for keyword in keywords if keyword in lowered)


def _normalize_ticket_status(status: str) -> str:
    aliases = {"resolved": "in_progress", "done": "in_progress", "closed": "pending_human", "waiting_customer": "waiting_on_customer", "pending_review": "pending_human"}
    normalized = (status or "").strip().lower()
    return normalized if normalized in {"open", "in_progress", "investigating", "monitoring", "waiting_on_customer", "pending_finance", "pending_human", "escalated"} else aliases.get(normalized, "in_progress")


def _fallback_reason(step: str, source: str, exc: Exception) -> str:
    return f"{step} 从 {source} 回退：{type(exc).__name__}"


def _extract_first_json_object(text: str) -> dict[str, object] | None:
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return None


def _has_required_keys(payload: dict[str, object] | None, required_keys: list[str]) -> bool:
    if payload is None:
        return False
    return all(key in payload for key in required_keys)


def _triage_payload_valid(payload: dict[str, object] | None) -> bool:
    if not _has_required_keys(payload, ["category", "priority", "urgency", "confidence", "reasoning"]):
        return False
    category = str(payload.get("category") or "").strip().lower()
    priority = str(payload.get("priority") or "").strip().lower()
    urgency = str(payload.get("urgency") or "").strip().lower()
    confidence = payload.get("confidence")
    if category not in ALLOWED_CATEGORIES:
        return False
    if priority not in ALLOWED_PRIORITIES:
        return False
    if urgency not in ALLOWED_URGENCIES:
        return False
    return isinstance(confidence, (int, float, str))


def _sla_review_payload_valid(payload: dict[str, object] | None) -> bool:
    if not _has_required_keys(payload, ["priority", "urgency", "sla_risk_score", "reasoning", "evidence"]):
        return False
    priority = str(payload.get("priority") or "").strip().lower()
    urgency = str(payload.get("urgency") or "").strip().lower()
    if priority not in ALLOWED_PRIORITIES:
        return False
    if urgency not in ALLOWED_URGENCIES:
        return False
    try:
        score = float(payload.get("sla_risk_score"))
    except (TypeError, ValueError):
        return False
    return 0.0 <= score <= 1.0


def _action_payload_valid(payload: dict[str, object] | None) -> bool:
    if not _has_required_keys(payload, ["action_type", "target_status", "requires_approval", "rationale", "confidence"]):
        return False
    action_type = str(payload.get("action_type") or "").strip().lower()
    target_status = _normalize_ticket_status(str(payload.get("target_status") or ""))
    if action_type not in ALLOWED_ACTIONS:
        return False
    if target_status not in ALLOWED_TARGET_STATUSES:
        return False
    return True


def _reply_payload_valid(payload: dict[str, object] | None) -> bool:
    if not _has_required_keys(payload, ["customer_reply", "internal_note", "status"]):
        return False
    status = _normalize_ticket_status(str(payload.get("status") or ""))
    return bool(str(payload.get("customer_reply") or "").strip()) and bool(str(payload.get("internal_note") or "").strip()) and status in ALLOWED_TARGET_STATUSES
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def _normalize_category(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {"refund": "billing_refund", "billing": "billing_refund", "payment": "billing_refund", "delivery": "delivery_issue", "shipping": "delivery_issue", "shipment": "delivery_issue", "logistics": "delivery_issue", "technical": "technical_issue", "incident": "technical_issue", "outage": "technical_issue", "bug": "technical_issue", "login": "account_access", "access": "account_access", "auth": "account_access", "general": "general_inquiry", "question": "general_inquiry", "inquiry": "general_inquiry"}
    normalized = aliases.get(raw, raw)
    return normalized if normalized in ALLOWED_CATEGORIES else "general_inquiry"


def _normalize_priority(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {"p0": "urgent", "critical": "urgent", "sev0": "urgent", "p1": "high", "major": "high", "sev1": "high", "p2": "medium", "normal": "medium", "moderate": "medium", "p3": "low", "minor": "low"}
    normalized = aliases.get(raw, raw)
    return normalized if normalized in ALLOWED_PRIORITIES else "medium"


def _normalize_urgency(value: object, priority: str) -> str:
    raw = str(value or "").strip().lower()
    aliases = {"critical": "sev1", "sev0": "sev1", "sev1": "sev1", "urgent": "same_day", "same_day": "same_day", "today": "same_day", "next_business_day": "next_business_day", "next day": "next_business_day", "standard": "standard", "normal": "standard"}
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
    aliases = {"refund_request": "refund", "issue_refund_request": "refund", "escalate": "escalation", "create_escalation": "escalation", "need_info": "request_info", "ask_for_info": "request_info", "update_status": "status_update", "debug": "troubleshoot", "investigate": "troubleshoot"}
    normalized = aliases.get(raw, raw)
    return normalized if normalized in ALLOWED_ACTIONS else "status_update"


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
    if normalized in ALLOWED_TARGET_STATUSES:
        return normalized
    return _default_target_status(action_type, category)


def _collect_top_citations(retrieved_docs: list[RetrievedDoc]) -> list[Citation]:
    return [citation for doc in retrieved_docs[:3] for citation in doc.citations[:1]]


def _policy_matches_action(doc: RetrievedDoc, action_type: str) -> bool:
    metadata_action = str(doc.metadata.get("action_type") or "").strip().lower()
    return metadata_action == action_type.lower() or action_type.lower() in f"{doc.title}\n{doc.snippet}".lower()


def _approval_required_from_policy(action_type: str, policy_hits: list[RetrievedDoc]) -> bool:
    return any(bool(doc.metadata.get("approval_required")) and _policy_matches_action(doc, action_type) for doc in policy_hits)


def _reply_alignment_ok(reply: str, action_type: str) -> bool:
    keyword_map = {"refund": ["退款", "审核", "财务"], "escalation": ["升级", "专家团队", "跟进", "值班"], "request_info": ["补充", "订单号", "截图", "信息"], "status_update": ["同步", "进展", "跟进", "核验"], "troubleshoot": ["排查", "日志", "报错", "影响范围"]}
    keywords = keyword_map.get(action_type, [])
    return True if not keywords else any(keyword in reply for keyword in keywords)


def _reply_policy_violation(reply: str, action_type: str, execution_status: str) -> bool:
    if action_type != "refund" and any(token in reply for token in ["退款成功", "已退款", "完成退款"]):
        return True
    if action_type == "request_info" and not any(token in reply for token in ["补充", "订单号", "截图", "信息"]):
        return True
    return execution_status != "executed" and any(token in reply for token in ["已解决", "已经修复", "处理完成"])


def _reply_hallucination(reply: str, ticket: TicketRecord, action_type: str, execution_status: str) -> bool:
    if ticket.linked_order_id:
        referenced = {item.upper() for item in re.findall(r"ORD-\d{4}", reply, flags=re.IGNORECASE)}
        if referenced and any(item != ticket.linked_order_id.upper() for item in referenced):
            return True
    if action_type != "refund" and any(token in reply for token in ["退款成功", "已退款", "退款已完成"]):
        return True
    return execution_status != "executed" and any(token in reply for token in ["已解决", "已经修复", "处理完成"])


@dataclass(slots=True)
class TriageAgent:
    primary_llm_client: OpenAICompatClient | None = None
    fallback_llm_client: OpenAICompatClient | None = None
    sla_risk_review_enabled: bool = True
    sla_risk_threshold: float = 0.72

    def analyze(self, ticket: TicketRecord) -> TriageResult:
        if self.primary_llm_client is None:
            return self._analyze_with_rules(ticket)
        try:
            return self._analyze_with_llm(ticket, self.primary_llm_client)
        except Exception as exc:
            primary_reason = _fallback_reason("分诊", self.primary_llm_client.source_label, exc)
            if self.fallback_llm_client is not None:
                result = self._analyze_with_llm(ticket, self.fallback_llm_client)
                result.fallback_reason = primary_reason
                return result
            raise RuntimeError(primary_reason) from exc

    def _repair_structured_payload(
        self,
        llm_client: OpenAICompatClient,
        *,
        task_name: str,
        invalid_payload: dict[str, object] | None,
        schema_hint: str,
    ) -> dict[str, object] | None:
        raw = llm_client.chat_text(
            "You repair invalid structured JSON for enterprise ticket workflows.",
            (
                f"task={task_name}\n"
                f"{schema_hint}\n"
                f"invalid_json={json.dumps(invalid_payload or {}, ensure_ascii=False)}\n"
                "Return JSON only."
            ),
        )
        return _extract_first_json_object(raw)

    def _analyze_with_rules(self, ticket: TicketRecord, fallback_reason: str | None = None) -> TriageResult:
        text = f"{ticket.title}\n{ticket.body}".lower()
        scores = {
            "billing_refund": _contains_any(text, ["refund", "return", "billing", "charge", "退款", "退货", "账单"]),
            "delivery_issue": _contains_any(text, ["shipping", "shipment", "delivery", "package", "发货", "物流", "快递", "包裹"]),
            "technical_issue": _contains_any(text, ["error", "outage", "production", "incident", "报错", "故障", "宕机", "生产", "无法使用"]),
            "account_access": _contains_any(text, ["login", "password", "mfa", "access", "reset", "登录", "密码", "验证码", "重置"]),
        }
        best = max(scores, key=scores.get)
        if scores[best] == 0:
            best = "general_inquiry"
        sla_risk = _contains_any(text, ["sla", "production", "urgent", "critical", "生产", "紧急", "严重", "阻塞"]) > 0
        urgency_signals = _contains_any(text, ["asap", "urgent", "impact", "affected", "尽快", "影响使用", "异常", "告警", "失败", "阻塞"])
        if best == "technical_issue" and (sla_risk or ticket.customer_tier == "enterprise"):
            priority, urgency = "urgent", "sev1"
        elif best == "technical_issue":
            priority, urgency = "high", "same_day"
        elif best == "delivery_issue":
            priority = "high" if ticket.customer_tier == "enterprise" or urgency_signals >= 2 else "medium"
            urgency = "same_day" if priority == "high" else "next_business_day"
        elif best == "account_access":
            priority = "high" if ticket.customer_tier == "enterprise" or urgency_signals >= 2 else "medium"
            urgency = "same_day" if priority == "high" else "next_business_day"
        elif best == "billing_refund":
            priority, urgency = "medium", "next_business_day"
        else:
            priority = "high" if ticket.customer_tier == "enterprise" or urgency_signals >= 2 else "medium" if urgency_signals == 1 else "low"
            urgency = "same_day" if priority == "high" else "next_business_day" if priority == "medium" else "standard"
        return TriageResult(
            category=best,
            priority=priority,
            urgency=urgency,
            sla_risk=sla_risk,
            sla_risk_score=0.85 if sla_risk else 0.15,
            sla_risk_reasoning="规则路径根据显式生产/SLA风险信号给出风险判断。",
            confidence=0.75,
            reasoning=f"规则判断为 {best}，优先级 {priority}。",
            decision_source="rule",
            fallback_reason=fallback_reason,
        )

    def _analyze_with_llm(self, ticket: TicketRecord, llm_client: OpenAICompatClient) -> TriageResult:
        prompt = (
            "Return JSON only with keys category, priority, urgency, confidence, reasoning. "
            "Allowed category: billing_refund, delivery_issue, technical_issue, account_access, general_inquiry. "
            "Allowed priority: low, medium, high, urgent. "
            "Use priority=urgent only when the text explicitly indicates SLA risk, contractual breach, or production impact with time-critical business interruption. "
            "For technical_issue, use priority=high when many users or a whole team are blocked, and urgent only when the text clearly says production is affected, SLA is at risk, or a contractual service deadline may be missed. "
            "For account_access, use priority=medium for normal single-user login, password reset, MFA, credential, or verification issues; use high when the text explicitly says daily operations are blocked or an important account cannot work, and urgent only with explicit SLA/production impact. "
            "For billing_refund, use priority=medium by default; use high when the text explicitly describes repeated incorrect charging, significant financial impact, or a strong urgency statement tied to business handling. "
            "For delivery_issue, use priority=high when the shipment has stalled for many days, tracking has not updated for a long time, or the customer explicitly asks urgent follow-up or escalation; otherwise use medium. "
            "For general_inquiry such as product capability, pricing, launch plan, or business consultation, use priority=low unless the text clearly says the current business is blocked or there is immediate time pressure beyond a normal request for quick reply. "
            "Use account_access for login, MFA, password-reset, or sign-in failures. "
            "Do not output SLA risk here; a second model will review it separately. "
            f"customer_tier={ticket.customer_tier}\nchannel={ticket.channel}\nproduct={ticket.product}\n"
            f"linked_order_id={ticket.linked_order_id or 'none'}\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}"
        )
        if llm_client.source_label == "llm_cloud":
            payload = llm_client.chat_json("You are a ticket triage model.", prompt)
        else:
            raw = llm_client.chat_text("You are a ticket triage model.", prompt)
            payload = _extract_first_json_object(raw)
        if payload is None:
            raise ValueError("LLM 未返回可解析的分诊 JSON")
        payload["category"] = _normalize_category(payload.get("category"))
        payload["priority"] = _normalize_priority(payload.get("priority"))
        payload["urgency"] = _normalize_urgency(payload.get("urgency"), str(payload["priority"]))
        payload["sla_risk"] = False
        payload["sla_risk_score"] = 0.0
        payload["confidence"] = _normalize_confidence(payload.get("confidence"), 0.65)
        payload["reasoning"] = str(payload.get("reasoning") or "LLM completed ticket triage.")
        result = TriageResult.model_validate(payload)
        if self.sla_risk_review_enabled:
            review = self._review_priority_and_sla_risk(ticket, result, llm_client)
            result.sla_risk_score = _normalize_confidence(review.get("sla_risk_score"), 0.0)
            result.sla_risk = result.sla_risk_score >= self.sla_risk_threshold
            if result.sla_risk:
                final_priority = "urgent"
            elif result.priority == "urgent":
                final_priority = "high"
            else:
                final_priority = result.priority
            result.priority = final_priority
            result.urgency = "sev1" if result.priority == "urgent" else _normalize_urgency(review.get("urgency"), result.priority)
            result.sla_risk_reasoning = str(review.get("reasoning") or "LLM completed SLA risk review.")
            evidence = review.get("evidence") or []
            result.sla_risk_evidence = [str(item) for item in evidence][:3]
        result.decision_source = llm_client.source_label  # type: ignore[assignment]
        result.fallback_reason = None
        return result

    def _review_priority_and_sla_risk(
        self,
        ticket: TicketRecord,
        preliminary: TriageResult,
        llm_client: OpenAICompatClient,
    ) -> dict[str, object]:
        prompt = (
            "Return JSON only with keys priority, urgency, sla_risk_score, reasoning, evidence. "
            "You are reviewing only ticket priority and SLA risk. "
            "sla_risk_score must be a number between 0 and 1. "
            "Score >= 0.80 only when the ticket explicitly states SLA risk, contractual deadline breach, or production impact together with service-commitment language. "
            "Scores between 0.45 and 0.75 mean the issue is high-impact or urgent, but evidence for a true SLA breach is incomplete. "
            "Scores <= 0.40 should be used for normal billing, logistics, general inquiries, and single-user account issues without explicit SLA/contract language. "
            "Do not treat 'many users affected', 'team work blocked', 'security incident', 'data breach', or 'needs escalation' alone as enough evidence for SLA risk unless the text also states production impact with SLA/contract pressure. "
            "Use priority=urgent only for explicit SLA/contract breach or production-impact incidents with time-critical commitment risk. Use priority=high for severe incidents that are important but do not clearly imply SLA breach. "
            "Use urgency=sev1 only for urgent cases with explicit SLA/contract risk; otherwise severe technical issues should generally stay at same_day. "
            "Return up to three short evidence strings copied or paraphrased from the ticket. "
            f"preliminary={preliminary.model_dump(mode='json')}\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}\n"
            f"title_source={ticket.source_subject or ''}\nbody_source={ticket.source_body or ''}"
        )
        if llm_client.source_label == "llm_cloud":
            payload = llm_client.chat_json("You are the SLA-risk review model in an enterprise ticket workflow.", prompt)
        else:
            raw = llm_client.chat_text("You are the SLA-risk review model in an enterprise ticket workflow.", prompt)
            payload = _extract_first_json_object(raw)
        if payload is None:
            raise ValueError("LLM 未返回可解析的 SLA 风险复核 JSON")
        return payload


@dataclass(slots=True)
class KnowledgeAgent:
    def retrieve(self, customer: CustomerProfile | None, order: OrderRecord | None, rag_docs: list[RetrievedDoc]) -> list[RetrievedDoc]:
        docs = list(rag_docs)
        if customer is not None:
            docs.append(RetrievedDoc(doc_id=f"customer:{customer.customer_id}", source_type="customer", title=f"{customer.name} 的客户画像", snippet=f"客户等级={customer.customer_tier}，地区={customer.region}，合作年限={customer.loyalty_years}，未关闭工单={customer.open_tickets}", score=1.0, metadata=customer.model_dump(mode="json"), citations=[Citation(source_path=f"db:customers/{customer.customer_id}", note="客户画像")]))
        if order is not None:
            docs.append(RetrievedDoc(doc_id=f"order:{order.order_id}", source_type="order", title=f"订单 {order.order_id} 状态", snippet=f"状态={order.status}，签收天数={order.delivered_days_ago}，可退款={order.eligible_for_refund}，金额={order.amount}", score=1.0, metadata=order.model_dump(mode="json"), citations=[Citation(source_path=f"db:orders/{order.order_id}", note="订单记录")]))
        return docs


@dataclass(slots=True)
class ResolutionAgent:
    action_primary_llm_client: OpenAICompatClient | None = None
    action_fallback_llm_client: OpenAICompatClient | None = None
    draft_primary_llm_client: OpenAICompatClient | None = None
    draft_fallback_llm_client: OpenAICompatClient | None = None
    reply_review_enabled: bool = True
    reply_supported_claim_threshold: float = 0.7

    def preview_action(self, ticket: TicketRecord, triage: TriageResult, customer: CustomerProfile | None, order: OrderRecord | None, policy_hits: list[RetrievedDoc], retrieved_docs: list[RetrievedDoc]) -> ActionProposal:
        return self._propose_action_with_rules(ticket, triage, customer, order, policy_hits, retrieved_docs)

    def propose_action(self, ticket: TicketRecord, triage: TriageResult, customer: CustomerProfile | None, order: OrderRecord | None, policy_hits: list[RetrievedDoc], retrieved_docs: list[RetrievedDoc]) -> ActionProposal:
        if self.action_primary_llm_client is None:
            return self._propose_action_with_rules(ticket, triage, customer, order, policy_hits, retrieved_docs)
        try:
            return self._propose_action_with_llm(self.action_primary_llm_client, ticket, triage, customer, order, policy_hits, retrieved_docs)
        except Exception as exc:
            primary_reason = _fallback_reason("动作建议", self.action_primary_llm_client.source_label, exc)
            if self.action_fallback_llm_client is not None:
                result = self._propose_action_with_llm(self.action_fallback_llm_client, ticket, triage, customer, order, policy_hits, retrieved_docs)
                result.fallback_reason = primary_reason
                return result
            raise RuntimeError(primary_reason) from exc

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
        decision_source: str = "rule",
        fallback_reason: str | None = None,
    ) -> ActionProposal:
        resolved_target_status = _normalize_target_status(target_status, action_type, triage.category)
        if action_type == "refund":
            tool_name, tool_args = "issue_refund_request", {"ticket_id": ticket.ticket_id, "order_id": order.order_id if order else ticket.linked_order_id, "amount": order.amount if order else None, "rationale": "订单满足退款条件，进入退款审批流程。"}
        elif action_type == "escalation":
            tool_name, tool_args = "create_escalation", {"ticket_id": ticket.ticket_id, "reason": "当前问题需要交由专家团队继续处理。", "priority": triage.priority}
        else:
            tool_name, tool_args = "update_ticket_status", {"ticket_id": ticket.ticket_id, "status": resolved_target_status}
        resolved_requires_approval = bool(requires_approval) if requires_approval is not None else action_type in {"refund", "escalation", "sla_override", "close_ticket_without_contact"}
        return ActionProposal(
            action_type=action_type,
            target_status=resolved_target_status,
            rationale=rationale,
            requires_approval=resolved_requires_approval,
            suggested_tool=tool_name,
            tool_args=tool_args,
            confidence=round(confidence, 2),
            decision_source=decision_source,
            fallback_reason=fallback_reason,
        )

    def _propose_action_with_rules(self, ticket: TicketRecord, triage: TriageResult, customer: CustomerProfile | None, order: OrderRecord | None, policy_hits: list[RetrievedDoc], retrieved_docs: list[RetrievedDoc], fallback_reason: str | None = None) -> ActionProposal:
        del customer, retrieved_docs
        if triage.category == "billing_refund":
            if not ticket.linked_order_id or order is None:
                return self._build_action_proposal(action_type="request_info", ticket=ticket, triage=triage, order=order, rationale="缺少可核验的订单信息，需要客户补充信息。", confidence=0.78, target_status="waiting_on_customer", requires_approval=False, decision_source="rule", fallback_reason=fallback_reason)
            if order.eligible_for_refund:
                return self._build_action_proposal(action_type="refund", ticket=ticket, triage=triage, order=order, rationale="订单满足退款条件，建议进入退款审批。", confidence=0.88, target_status="pending_finance", requires_approval=True, decision_source="rule", fallback_reason=fallback_reason)
            return self._build_action_proposal(action_type="status_update", ticket=ticket, triage=triage, order=order, rationale="订单当前不满足退款条件，先同步规则限制。", confidence=0.70, target_status="in_progress", requires_approval=False, decision_source="rule", fallback_reason=fallback_reason)
        if triage.category == "technical_issue":
            action_type = "escalation" if triage.priority in {"high", "urgent"} or ticket.customer_tier == "enterprise" else "troubleshoot"
            return self._build_action_proposal(action_type=action_type, ticket=ticket, triage=triage, order=order, rationale="技术问题需要排障或升级处理。", confidence=0.84, target_status=_default_target_status(action_type, triage.category), requires_approval=(action_type == "escalation"), decision_source="rule", fallback_reason=fallback_reason)
        if triage.category == "delivery_issue":
            action_type = "escalation" if order is not None and (order.status == "pending" or (order.delivered_days_ago or 0) > 5) else "status_update"
            return self._build_action_proposal(action_type=action_type, ticket=ticket, triage=triage, order=order, rationale="物流问题需要同步状态或升级。", confidence=0.75, target_status=_default_target_status(action_type, triage.category), requires_approval=(action_type == "escalation"), decision_source="rule", fallback_reason=fallback_reason)
        if triage.category == "account_access":
            return self._build_action_proposal(action_type="troubleshoot", ticket=ticket, triage=triage, order=order, rationale="账号访问问题优先排障。", confidence=0.80, target_status="investigating", requires_approval=False, decision_source="rule", fallback_reason=fallback_reason)
        return self._build_action_proposal(action_type="status_update", ticket=ticket, triage=triage, order=order, rationale="一般咨询先同步进展。", confidence=0.66, target_status=_default_target_status("status_update", triage.category), requires_approval=False, decision_source="rule", fallback_reason=fallback_reason)

    def _propose_action_with_llm(self, llm_client: OpenAICompatClient, ticket: TicketRecord, triage: TriageResult, customer: CustomerProfile | None, order: OrderRecord | None, policy_hits: list[RetrievedDoc], retrieved_docs: list[RetrievedDoc]) -> ActionProposal:
        evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:6])
        prompt = (
            "Return JSON only with keys action_type, target_status, requires_approval, rationale, confidence. "
            "Allowed action_type: refund, escalation, request_info, status_update, troubleshoot. "
            "Allowed target_status: in_progress, investigating, monitoring, waiting_on_customer, pending_finance, pending_human, escalated. "
            "If the issue is a production outage, enterprise-impact incident, or severe technical blocker, prefer escalation. "
            "For normal account_access issues such as login, MFA, password reset, or single-user sign-in failure, prefer troubleshoot instead of escalation. "
            "If refund evidence is incomplete or no valid order exists, prefer request_info instead of refund. "
            "If order is present and eligible_for_refund=true, prefer refund rather than request_info unless the ticket explicitly says information is still missing. "
            "For general inquiries asking for explanation, clarification, or progress update, prefer status_update instead of request_info. "
            "Use request_info only when the current ticket truly lacks essential identifiers or proof needed to continue processing. "
            "Set requires_approval=true for refund and escalation, and for any action that changes financial outcome or major customer status. "
            "If action_type=refund, target_status should usually be pending_finance. "
            "If action_type=escalation, target_status should usually be escalated. "
            "If action_type=request_info, target_status should usually be waiting_on_customer. "
            "If action_type=troubleshoot, target_status should usually be investigating. "
            "If action_type=status_update, target_status should usually be in_progress or monitoring. "
            "Example A: single-user login/MFA/password issue -> action_type=troubleshoot, target_status=investigating, requires_approval=false. "
            "Example B: valid delivered order with eligible_for_refund=true and explicit refund request -> action_type=refund, target_status=pending_finance, requires_approval=true. "
            "Example C: general inquiry asking for progress or clarification -> action_type=status_update, target_status=in_progress, requires_approval=false. "
            "Do not invent tool outputs or policy hits.\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}\n"
            f"title_source={ticket.source_subject or ''}\nbody_source={ticket.source_body or ''}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"customer={customer.model_dump(mode='json') if customer else 'null'}\n"
            f"order={order.model_dump(mode='json') if order else 'null'}\n"
            f"policies={[doc.snippet for doc in policy_hits[:5]]}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        if llm_client.source_label == "llm_cloud":
            payload = llm_client.chat_json("You are an enterprise ticket action-decision model.", prompt)
        else:
            raw = llm_client.chat_text("You are an enterprise ticket action-decision model.", prompt)
            payload = _extract_first_json_object(raw)
        if payload is None:
            raise ValueError("LLM 未返回可解析的动作建议 JSON")
        action_type = _normalize_action_type(payload.get("action_type"))
        target_status = _normalize_target_status(payload.get("target_status"), action_type, triage.category)
        return self._build_action_proposal(
            action_type=action_type,
            ticket=ticket,
            triage=triage,
            order=order,
            rationale=str(payload.get("rationale") or "LLM generated the next action proposal."),
            confidence=_normalize_confidence(payload.get("confidence"), 0.68),
            target_status=target_status,
            requires_approval=_normalize_bool(payload.get("requires_approval")),
            decision_source=llm_client.source_label,
        )

    def draft_reply(self, ticket: TicketRecord, triage: TriageResult, proposed_action: ActionProposal, execution_result: object, retrieved_docs: list[RetrievedDoc], templates: list[str], review_decision: ReviewDecision | None = None) -> TicketResponse:
        if self.draft_primary_llm_client is None:
            return self._draft_with_rules(ticket, triage, proposed_action, execution_result, retrieved_docs, templates, review_decision)
        try:
            generated = self._draft_with_llm(self.draft_primary_llm_client, ticket, triage, proposed_action, execution_result, retrieved_docs, review_decision)
            if not self.reply_review_enabled:
                return generated
            verifier_client = (
                self.draft_fallback_llm_client
                if self.draft_primary_llm_client.source_label == "llm_minimind" and self.draft_fallback_llm_client is not None
                else self.draft_primary_llm_client
            )
            review_payload = self._review_reply_with_llm(
                verifier_client,
                ticket,
                triage,
                proposed_action,
                execution_result,
                retrieved_docs,
                generated,
            )
            if not review_payload["review_passed"]:
                if self.draft_primary_llm_client.source_label == "llm_minimind":
                    raise ValueError("MiniMind 回复未通过事实校验")
                generated = self._repair_reply_with_llm(
                    self.draft_primary_llm_client,
                    ticket,
                    triage,
                    proposed_action,
                    execution_result,
                    retrieved_docs,
                    generated,
                    review_payload,
                )
                review_payload = self._review_reply_with_llm(
                    verifier_client,
                    ticket,
                    triage,
                    proposed_action,
                    execution_result,
                    retrieved_docs,
                    generated,
                )
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

    def _draft_with_rules(self, ticket: TicketRecord, triage: TriageResult, proposed_action: ActionProposal, execution_result: object, retrieved_docs: list[RetrievedDoc], templates: list[str], review_decision: ReviewDecision | None = None, fallback_reason: str | None = None) -> TicketResponse:
        del triage, review_decision
        template = templates[0] if templates else "我们已经收到你的诉求，正在根据当前证据推进处理。"
        if proposed_action.action_type == "refund":
            customer_reply, internal_note, status = f"我们已经根据订单 {ticket.linked_order_id or '待补充'} 发起退款审核，财务确认后会继续同步进展。", "已发起退款审批，请持续跟进财务状态。", "pending_finance"
        elif proposed_action.action_type == "escalation":
            customer_reply, internal_note, status = "当前问题已升级给专家团队处理，我们会基于现有上下文继续推进，并尽快反馈。", "升级单已创建，建议值班负责人优先跟进。", "escalated"
        elif proposed_action.action_type == "request_info":
            customer_reply, internal_note, status = "为了继续安全处理，请补充订单号、付款截图或其他可核验信息。", "等待客户补充关键信息后再继续处理。", "waiting_on_customer"
        elif proposed_action.action_type == "troubleshoot":
            customer_reply, internal_note, status = "我们已经开始排查当前问题，会优先核对报错信息、影响范围和临时绕过方案。", "进入标准排障流程，建议继续补充日志和环境信息。", "investigating"
        else:
            customer_reply, internal_note, status = template, "继续跟进当前工单，并同步阶段性进展。", "monitoring" if proposed_action.action_type == "status_update" and ticket.expected_category == "delivery_issue" else "in_progress"
        if getattr(execution_result, "status", None) in {"needs_handoff", "rejected"}:
            internal_note, status = f"{internal_note} 当前自动流程已中止，建议人工接手。", "pending_human"
        return TicketResponse(customer_reply=customer_reply, internal_note=internal_note, status=_normalize_ticket_status(status), citations=_collect_top_citations(retrieved_docs), draft_source="rule", fallback_reason=fallback_reason)

    def _draft_with_llm(self, llm_client: OpenAICompatClient, ticket: TicketRecord, triage: TriageResult, proposed_action: ActionProposal, execution_result: object, retrieved_docs: list[RetrievedDoc], review_decision: ReviewDecision | None = None) -> TicketResponse:
        execution_payload = getattr(execution_result, "__dict__", execution_result)
        evidence = "\n".join(f"- [{doc.source_type}] {doc.title}: {doc.snippet}" for doc in retrieved_docs[:6])
        prompt = (
            "Return JSON only with keys customer_reply, internal_note, status. "
            "Write customer_reply and internal_note in Chinese. "
            "Never claim a refund, fix, or delivery completion unless execution.status is executed.\n"
            f"title_cn={ticket.title}\nbody_cn={ticket.body}\n"
            f"title_source={ticket.source_subject or ''}\nbody_source={ticket.source_body or ''}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"action={proposed_action.model_dump(mode='json')}\n"
            f"execution={execution_payload}\n"
            f"review={review_decision.model_dump(mode='json') if review_decision else 'null'}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        if llm_client.source_label == "llm_cloud":
            payload = llm_client.chat_json("You are the reply-generation model in an enterprise ticket workflow.", prompt)
        else:
            raw = llm_client.chat_text("You are the reply-generation model in an enterprise ticket workflow.", prompt)
            payload = _extract_first_json_object(raw)
        if payload is None:
            raise ValueError("LLM 未返回可解析的回复 JSON")
        payload.setdefault("status", "in_progress")
        payload.setdefault("internal_note", "LLM generated a structured reply.")
        response = TicketResponse.model_validate(payload)
        response.status = _normalize_ticket_status(response.status)
        response.citations = _collect_top_citations(retrieved_docs)
        response.draft_source = llm_client.source_label  # type: ignore[assignment]
        response.fallback_reason = None
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
            "You are verifying whether the customer_reply is fully supported by the retrieved evidence and execution result. "
            "supported_claim_ratio must be a number between 0 and 1. "
            "Mark hallucination_risk=true if the reply invents an order id, refund completion, fix completion, or unsupported delivery status. "
            "Mark policy_safe=false if the reply violates the action intent, skips required customer information, or makes a hard commitment not present in evidence. "
            "needs_rewrite=true when grounded=false, policy_safe=false, hallucination_risk=true, or supported_claim_ratio < 0.70. "
            f"ticket={ticket.model_dump(mode='json')}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"action={proposed_action.model_dump(mode='json')}\n"
            f"execution={execution_payload}\n"
            f"reply={response.model_dump(mode='json')}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        if llm_client.source_label == "llm_cloud":
            payload = llm_client.chat_json("You are a grounded-reply verifier in an enterprise ticket workflow.", prompt)
        else:
            raw = llm_client.chat_text("You are a grounded-reply verifier in an enterprise ticket workflow.", prompt)
            payload = _extract_first_json_object(raw)
        if payload is None:
            raise ValueError("LLM 未返回可解析的回复校验 JSON")
        grounded = _normalize_bool(payload.get("grounded"))
        policy_safe = _normalize_bool(payload.get("policy_safe"))
        hallucination_risk = _normalize_bool(payload.get("hallucination_risk"))
        supported_claim_ratio = _normalize_confidence(payload.get("supported_claim_ratio"), 0.0)
        needs_rewrite = _normalize_bool(payload.get("needs_rewrite")) or supported_claim_ratio < self.reply_supported_claim_threshold
        review_note = str(payload.get("review_note") or "LLM completed reply verification.")
        return {
            "grounded": grounded,
            "policy_safe": policy_safe,
            "hallucination_risk": hallucination_risk,
            "supported_claim_ratio": supported_claim_ratio,
            "review_note": review_note,
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
            "Rewrite the reply to remove unsupported claims and keep it conservative, operational, and evidence-grounded. "
            "Do not invent fixes, refunds, or delivery completion. "
            f"ticket={ticket.model_dump(mode='json')}\n"
            f"triage={triage.model_dump(mode='json')}\n"
            f"action={proposed_action.model_dump(mode='json')}\n"
            f"execution={execution_payload}\n"
            f"original_reply={response.model_dump(mode='json')}\n"
            f"review_note={review_payload.get('review_note')}\n"
            f"evidence=\n{evidence or '- none'}"
        )
        if llm_client.source_label == "llm_cloud":
            payload = llm_client.chat_json("You are a reply-repair model in an enterprise ticket workflow.", prompt)
        else:
            raw = llm_client.chat_text("You are a reply-repair model in an enterprise ticket workflow.", prompt)
            payload = _extract_first_json_object(raw)
        if payload is None:
            raise ValueError("LLM 未返回可解析的回复修复 JSON")
        payload.setdefault("status", response.status)
        payload.setdefault("internal_note", response.internal_note)
        repaired = TicketResponse.model_validate(payload)
        repaired.status = _normalize_ticket_status(repaired.status)
        repaired.citations = response.citations
        repaired.draft_source = response.draft_source
        repaired.fallback_reason = response.fallback_reason
        return repaired
