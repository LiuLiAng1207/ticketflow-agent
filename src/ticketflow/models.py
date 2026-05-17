from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field
from typing_extensions import NotRequired, TypedDict


Priority = Literal["low", "medium", "high", "urgent"]
RiskLevel = Literal["standard", "elevated", "high"]
Category = Literal[
    "billing_refund",
    "delivery_issue",
    "technical_issue",
    "account_access",
    "general_inquiry",
]
ActionRouteFamily = Literal["refund_candidate", "escalation_candidate", "standard_resolution"]
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
ApprovalMode = Literal["auto", "approve_reject", "approve_edit_reject"]
DecisionSource = Literal["rule", "governance", "llm_cloud", "llm_minimind", "human_edit"]


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


class AttachmentRecord(BaseModel):
    attachment_id: str
    ticket_id: str
    filename: str
    file_type: Literal["image", "pdf", "text", "log", "unknown"] = "unknown"
    source_dataset: str | None = None
    storage_path: str | None = None
    content_hash: str | None = None
    ocr_text: str = ""
    visual_summary: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    parse_status: Literal["pending", "parsed", "failed"] = "pending"
    created_at: datetime | None = None


class AttachmentEvidence(BaseModel):
    evidence_id: str
    attachment_id: str
    ticket_id: str
    evidence_type: str
    extracted_text: str = ""
    visual_summary: str = ""
    entities: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.0
    source_span: str | None = None
    bbox: list[float] | None = None
    risk_flags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedDoc(BaseModel):
    doc_id: str
    source_type: Literal["kb", "policy", "history", "customer", "order", "template", "attachment"]
    title: str
    snippet: str
    score: float = 0.0
    rerank_score: float | None = None
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
    metadata: dict[str, Any] = Field(default_factory=dict)


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
    route_family: ActionRouteFamily = "standard_resolution"
    sufficiency_passed: bool = True
    supporting_doc_ids: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    decision_source: DecisionSource = "rule"
    fallback_reason: str | None = None


class ContextSufficiencyResult(BaseModel):
    route_family: ActionRouteFamily
    risk_level: RiskLevel
    required_sources: list[str] = Field(default_factory=list)
    supporting_doc_ids: list[str] = Field(default_factory=list)
    supporting_history_doc_ids: list[str] = Field(default_factory=list)
    missing_sources: list[str] = Field(default_factory=list)
    sufficient: bool
    fallback_route: ActionType | None = None
    support_strength: float = 0.0
    reasoning: str
    decision_source: DecisionSource = "rule"
    fallback_reason: str | None = None


class ActionRoutingDecision(BaseModel):
    route_family: ActionRouteFamily
    sufficient: bool
    allowed_actions: list[ActionType] = Field(default_factory=list)
    forced_action_type: ActionType | None = None
    forced_target_status: str | None = None
    reasoning: str


class ToolApprovalPolicy(BaseModel):
    tool_name: str
    sensitive: bool
    approval_mode: ApprovalMode = "auto"
    requires_sufficiency: bool = False
    allowed_decisions: list[ApprovalDecision] = Field(default_factory=list)
    description: str = ""


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


class ReplyFactCheckResult(BaseModel):
    grounded: bool
    policy_safe: bool
    hallucination_risk: bool
    supported_claim_ratio: float
    unsupported_claims: list[str] = Field(default_factory=list)
    rewrite_needed: bool = False
    review_note: str = ""
    passed: bool = True
    rewritten: bool = False
    downgraded: bool = False


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
    attachments: NotRequired[list[AttachmentRecord]]
    attachment_evidence: NotRequired[list[AttachmentEvidence]]
    triage_result: NotRequired[TriageResult | None]
    retrieved_docs: NotRequired[list[RetrievedDoc]]
    retrieval_stats: NotRequired[RetrievalStats | None]
    policy_hits: NotRequired[list[RetrievedDoc]]
    sufficiency_result: NotRequired[ContextSufficiencyResult | None]
    action_route: NotRequired[ActionRoutingDecision | None]
    proposed_action: NotRequired[ActionProposal | None]
    tool_approval_policy: NotRequired[ToolApprovalPolicy | None]
    review_decision: NotRequired[ReviewDecision | None]
    execution_result: NotRequired[ExecutionResult | None]
    reply_draft: NotRequired[TicketResponse | None]
    reply_fact_check: NotRequired[ReplyFactCheckResult | None]
    draft_reply: NotRequired[TicketResponse | None]
    approval_state: NotRequired[str]
    audit_log: NotRequired[list[AuditEvent]]
    trace: NotRequired[list[dict[str, Any]]]
    tool_calls: NotRequired[list[dict[str, Any]]]
    external_ops: NotRequired[list[ExternalOpRecord]]
