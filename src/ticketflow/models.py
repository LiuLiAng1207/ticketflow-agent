from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


Priority = Literal["low", "medium", "high", "urgent"]
Category = Literal[
    "billing_refund",
    "delivery_issue",
    "technical_issue",
    "account_access",
    "general_inquiry",
]
ActionType = Literal[
    "refund",
    "escalation",
    "request_info",
    "status_update",
    "troubleshoot",
    "sla_override",
    "close_ticket_without_contact",
]
ApprovalDecision = Literal["approve", "edit", "reject"]
DecisionSource = Literal["rule", "llm_cloud", "llm_minimind", "human_edit"]


class Citation(BaseModel):
    source_path: str
    line_start: int | None = None
    line_end: int | None = None
    page: int | None = None
    note: str | None = None


class TicketRecord(BaseModel):
    ticket_id: str
    channel: str
    customer_id: str
    customer_tier: str
    title: str
    body: str
    product: str
    created_at: datetime
    status: str
    linked_order_id: str | None = None
    expected_category: Category | None = None
    source_dataset: str | None = None
    source_ticket_ref: str | None = None
    source_language: str | None = None
    source_queue: str | None = None
    source_subject: str | None = None
    source_body: str | None = None


class CustomerProfile(BaseModel):
    customer_id: str
    name: str
    email: str
    customer_tier: str
    region: str
    loyalty_years: int
    open_tickets: int


class OrderRecord(BaseModel):
    order_id: str
    customer_id: str
    product: str
    status: str
    delivered_days_ago: int | None = None
    amount: float
    eligible_for_refund: bool


class RetrievedDoc(BaseModel):
    doc_id: str
    source_type: Literal["kb", "policy", "history", "customer", "order", "template"]
    title: str
    snippet: str
    score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)


class RetrievalStats(BaseModel):
    lexical_hits: int = 0
    vector_hits: int = 0
    final_hits: int = 0
    used_vector: bool = False
    fallback_to_keywords: bool = False
    source_counts: dict[str, int] = Field(default_factory=dict)
    retrieved_doc_ids: list[str] = Field(default_factory=list)


class TriageResult(BaseModel):
    category: Category
    priority: Priority
    urgency: str
    sla_risk: bool
    sla_risk_score: float = 0.0
    sla_risk_reasoning: str | None = None
    sla_risk_evidence: list[str] = Field(default_factory=list)
    confidence: float
    reasoning: str
    decision_source: DecisionSource = "rule"
    fallback_reason: str | None = None


class ActionProposal(BaseModel):
    action_type: ActionType
    target_status: str = "in_progress"
    rationale: str
    requires_approval: bool
    suggested_tool: str
    tool_args: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    decision_source: DecisionSource = "rule"
    fallback_reason: str | None = None


class ReviewDecision(BaseModel):
    decision: ApprovalDecision
    edited_action: ActionProposal | None = None
    comment: str | None = None


class ExecutionResult(BaseModel):
    status: Literal["executed", "skipped", "needs_handoff", "rejected"]
    tool_name: str | None = None
    tool_output: dict[str, Any] = Field(default_factory=dict)
    handoff_required: bool = False
    error: str | None = None


class AuditEvent(BaseModel):
    timestamp: datetime
    actor: str
    event_type: str
    detail: str
    payload: dict[str, Any] = Field(default_factory=dict)


class TicketResponse(BaseModel):
    customer_reply: str
    internal_note: str
    status: str
    citations: list[Citation] = Field(default_factory=list)
    review_passed: bool = True
    review_grounded: bool | None = None
    review_policy_safe: bool | None = None
    review_hallucination_risk: bool | None = None
    review_supported_claim_ratio: float | None = None
    review_note: str | None = None
    draft_source: DecisionSource = "rule"
    fallback_reason: str | None = None


class ExternalOpRecord(BaseModel):
    op_type: Literal["incident_email", "kb_candidate_email"]
    status: Literal["sent", "failed", "skipped"]
    recipient: str
    subject: str
    provider_message_id: str | None = None
    delivery_id: str | None = None
    latency_ms: int | None = None
    error_message: str | None = None
    created_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


class InvocationResult(BaseModel):
    state: dict[str, Any]
    interrupted: bool = False
    interrupt_payload: dict[str, Any] | None = None


class TicketFlowState(TypedDict):
    thread_id: str
    ticket: TicketRecord
    triage_result: NotRequired[TriageResult | None]
    retrieved_docs: NotRequired[list[RetrievedDoc]]
    retrieval_stats: NotRequired[RetrievalStats | None]
    policy_hits: NotRequired[list[RetrievedDoc]]
    proposed_action: NotRequired[ActionProposal | None]
    review_decision: NotRequired[ReviewDecision | None]
    execution_result: NotRequired[ExecutionResult | None]
    draft_reply: NotRequired[TicketResponse | None]
    approval_state: NotRequired[str]
    audit_log: NotRequired[list[AuditEvent]]
    trace: NotRequired[list[dict[str, Any]]]
    tool_calls: NotRequired[list[dict[str, Any]]]
    external_ops: NotRequired[list[ExternalOpRecord]]
