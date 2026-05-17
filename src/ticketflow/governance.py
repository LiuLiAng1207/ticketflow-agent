from __future__ import annotations

from dataclasses import dataclass, field

from .models import ActionRoutingDecision, ActionRouteFamily, TriageResult, ToolApprovalPolicy, TicketRecord


@dataclass(slots=True)
class EvidenceRequirementProfile:
    route_family: ActionRouteFamily
    required_sources: list[str] = field(default_factory=list)
    optional_sources: list[str] = field(default_factory=list)
    min_history_strength: float = 0.0


EVIDENCE_REQUIREMENT_PROFILES: dict[ActionRouteFamily, EvidenceRequirementProfile] = {
    "refund_candidate": EvidenceRequirementProfile(
        route_family="refund_candidate",
        required_sources=["policy", "order"],
        optional_sources=["history"],
        min_history_strength=0.0,
    ),
    "escalation_candidate": EvidenceRequirementProfile(
        route_family="escalation_candidate",
        required_sources=["policy", "history"],
        optional_sources=["customer"],
        min_history_strength=0.45,
    ),
    "standard_resolution": EvidenceRequirementProfile(
        route_family="standard_resolution",
        required_sources=[],
        optional_sources=["kb", "policy", "history"],
        min_history_strength=0.0,
    ),
}


TOOL_APPROVAL_POLICIES: dict[str, ToolApprovalPolicy] = {
    "issue_refund_request": ToolApprovalPolicy(
        tool_name="issue_refund_request",
        sensitive=True,
        approval_mode="approve_reject",
        requires_sufficiency=True,
        allowed_decisions=["approve", "reject"],
        description="Refund requests are financially sensitive and require explicit approval with sufficient policy and order evidence.",
    ),
    "create_escalation": ToolApprovalPolicy(
        tool_name="create_escalation",
        sensitive=True,
        approval_mode="approve_edit_reject",
        requires_sufficiency=True,
        allowed_decisions=["approve", "edit", "reject"],
        description="Escalations are sensitive operational actions and require explicit approval with sufficient policy and history evidence.",
    ),
    "update_ticket_status": ToolApprovalPolicy(
        tool_name="update_ticket_status",
        sensitive=False,
        approval_mode="auto",
        requires_sufficiency=False,
        allowed_decisions=[],
        description="Routine ticket status updates are auto-executed.",
    ),
    "mcp.send_incident_email": ToolApprovalPolicy(
        tool_name="mcp.send_incident_email",
        sensitive=False,
        approval_mode="auto",
        requires_sufficiency=False,
        allowed_decisions=[],
        description="Incident notification email inherits the upstream escalation decision in V1.",
    ),
    "mcp.submit_kb_candidate_email": ToolApprovalPolicy(
        tool_name="mcp.submit_kb_candidate_email",
        sensitive=False,
        approval_mode="auto",
        requires_sufficiency=False,
        allowed_decisions=[],
        description="KB candidate email is non-blocking and does not require separate approval in V1.",
    ),
}


def infer_route_family(ticket: TicketRecord, triage: TriageResult) -> ActionRouteFamily:
    if triage.category == "billing_refund":
        return "refund_candidate"
    if triage.category == "technical_issue" and (
        triage.priority in {"high", "urgent"} or triage.sla_risk or ticket.customer_tier == "enterprise"
    ):
        return "escalation_candidate"
    if triage.category == "delivery_issue" and (
        triage.priority in {"high", "urgent"} or ticket.customer_tier == "enterprise"
    ):
        return "escalation_candidate"
    return "standard_resolution"


def infer_risk_level(ticket: TicketRecord, triage: TriageResult) -> str:
    if triage.priority == "urgent" or triage.sla_risk:
        return "high"
    if triage.priority == "high" or ticket.customer_tier == "enterprise":
        return "elevated"
    return "standard"


def get_requirement_profile(route_family: ActionRouteFamily) -> EvidenceRequirementProfile:
    return EVIDENCE_REQUIREMENT_PROFILES[route_family]


def get_tool_approval_policy(tool_name: str) -> ToolApprovalPolicy:
    return TOOL_APPROVAL_POLICIES.get(
        tool_name,
        ToolApprovalPolicy(
            tool_name=tool_name,
            sensitive=False,
            approval_mode="auto",
            requires_sufficiency=False,
            allowed_decisions=[],
            description="No explicit approval policy configured.",
        ),
    )


def build_action_route(sufficiency_result) -> ActionRoutingDecision:
    if sufficiency_result.route_family == "refund_candidate":
        if not sufficiency_result.sufficient:
            return ActionRoutingDecision(
                route_family="refund_candidate",
                sufficient=False,
                allowed_actions=["request_info", "status_update"],
                forced_action_type="request_info",
                forced_target_status="waiting_on_customer",
                reasoning="Refund path is missing critical evidence, so the workflow must request more information before taking a financial action.",
            )
        return ActionRoutingDecision(
            route_family="refund_candidate",
            sufficient=True,
            allowed_actions=["refund"],
            reasoning="Refund path has sufficient evidence, so the workflow enters a route-family constrained refund decision instead of reopening the full action space.",
        )

    if sufficiency_result.route_family == "escalation_candidate":
        if not sufficiency_result.sufficient:
            return ActionRoutingDecision(
                route_family="escalation_candidate",
                sufficient=False,
                allowed_actions=["status_update", "request_info"],
                forced_action_type="status_update",
                forced_target_status="pending_human",
                reasoning="Escalation path is missing critical evidence, so the workflow must defer to a human queue instead of issuing an aggressive escalation.",
            )
        return ActionRoutingDecision(
            route_family="escalation_candidate",
            sufficient=True,
            allowed_actions=["escalation"],
            reasoning="Escalation path has sufficient evidence, so the workflow enters a route-family constrained escalation decision instead of allowing lower-risk fallback actions.",
        )

    return ActionRoutingDecision(
        route_family="standard_resolution",
        sufficient=sufficiency_result.sufficient,
        allowed_actions=["request_info", "status_update", "troubleshoot"],
        reasoning="Standard-resolution path stays in constrained action generation without mandatory blocking evidence.",
    )
