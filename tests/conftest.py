from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from ticketflow.agents_llm import ResolutionAgent, TriageAgent
from ticketflow.graph import TicketFlowRunner
from ticketflow.models import (
    ActionProposal,
    ReplyFactCheckResult,
    TicketRecord,
    TicketResponse,
    TriageResult,
)
from ticketflow.seed import build_seed_dataset


@pytest.fixture(scope="session")
def ticketflow_project() -> Path:
    project_root = Path.cwd() / ".pytest_runtime" / "ticketflow"
    if project_root.exists():
        shutil.rmtree(project_root, ignore_errors=True)
    (project_root / "data" / "seed").mkdir(parents=True)
    build_seed_dataset(project_root / "data" / "seed")
    return project_root


def _clear_generated_runtime(project_root: Path) -> None:
    generated_dir = project_root / "data" / "generated"
    if generated_dir.exists():
        shutil.rmtree(generated_dir, ignore_errors=True)


@pytest.fixture()
def runner(ticketflow_project: Path) -> TicketFlowRunner:
    _clear_generated_runtime(ticketflow_project)
    runner = TicketFlowRunner.from_project_root(
        ticketflow_project,
        overrides={
            "RAG_EMBED_BACKEND": "hash",
        },
    )
    runner.reset_demo_data()
    yield runner
    runner.close()


@pytest.fixture()
def runner_factory(ticketflow_project: Path):
    created_runners: list[TicketFlowRunner] = []

    def _make(**overrides: str) -> TicketFlowRunner:
        _clear_generated_runtime(ticketflow_project)
        merged_overrides = {"RAG_EMBED_BACKEND": "hash", **(overrides or {})}
        runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides=merged_overrides)
        runner.reset_demo_data()
        created_runners.append(runner)
        return runner

    yield _make
    for runner in created_runners:
        runner.close()


def _triage_priority(ticket: TicketRecord) -> str:
    if ticket.expected_category == "technical_issue" and ticket.customer_tier == "enterprise":
        return "urgent"
    if ticket.expected_category in {"billing_refund", "technical_issue"}:
        return "high"
    if ticket.expected_category == "delivery_issue":
        return "medium"
    return "medium"


def _triage_urgency(priority: str) -> str:
    return {
        "low": "standard",
        "medium": "next_business_day",
        "high": "same_day",
        "urgent": "sev1",
    }[priority]


def _stub_analyze(self: TriageAgent, ticket: TicketRecord) -> TriageResult:
    category = ticket.expected_category or "general_inquiry"
    priority = _triage_priority(ticket)
    text = f"{ticket.title}\n{ticket.body}".lower()
    sla_risk = category == "technical_issue" and (
        ticket.customer_tier == "enterprise" or any(token in text for token in ("sla", "生产", "critical", "sev1"))
    )
    return TriageResult(
        category=category,  # type: ignore[arg-type]
        priority=priority,  # type: ignore[arg-type]
        urgency=_triage_urgency(priority),
        sla_risk=sla_risk,
        sla_risk_score=1.0 if sla_risk else 0.0,
        sla_risk_reasoning="test stub triage",
        sla_risk_evidence=[],
        confidence=0.95,
        reasoning="deterministic test triage",
        decision_source="llm_minimind",
    )


def _stub_propose_action(
    self: ResolutionAgent,
    ticket: TicketRecord,
    triage: TriageResult,
    customer,
    order,
    policy_hits,
    retrieved_docs,
    sufficiency_result,
    action_route,
) -> ActionProposal:
    del customer, policy_hits, retrieved_docs
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

    if sufficiency_result.route_family == "refund_candidate":
        if order is not None and order.eligible_for_refund:
            return self._build_action_proposal(
                action_type="refund",
                ticket=ticket,
                triage=triage,
                order=order,
                rationale="test stub refund path",
                confidence=0.9,
                target_status="pending_finance",
                requires_approval=True,
                route_family=sufficiency_result.route_family,
                sufficiency_passed=sufficiency_result.sufficient,
                supporting_doc_ids=sufficiency_result.supporting_doc_ids,
                missing_sources=sufficiency_result.missing_sources,
                decision_source="llm_minimind",
            )
        return self._build_action_proposal(
            action_type="status_update",
            ticket=ticket,
            triage=triage,
            order=order,
            rationale="test stub non-refundable refund candidate",
            confidence=0.8,
            target_status="in_progress",
            requires_approval=False,
            route_family=sufficiency_result.route_family,
            sufficiency_passed=sufficiency_result.sufficient,
            supporting_doc_ids=sufficiency_result.supporting_doc_ids,
            missing_sources=sufficiency_result.missing_sources,
            decision_source="llm_minimind",
        )

    if sufficiency_result.route_family == "escalation_candidate":
        return self._build_action_proposal(
            action_type="escalation",
            ticket=ticket,
            triage=triage,
            order=order,
            rationale="test stub escalation path",
            confidence=0.88,
            target_status="escalated",
            requires_approval=True,
            route_family=sufficiency_result.route_family,
            sufficiency_passed=sufficiency_result.sufficient,
            supporting_doc_ids=sufficiency_result.supporting_doc_ids,
            missing_sources=sufficiency_result.missing_sources,
            decision_source="llm_minimind",
        )

    action_type = "troubleshoot" if triage.category in {"technical_issue", "account_access"} else "status_update"
    target_status = "investigating" if action_type == "troubleshoot" else ("monitoring" if triage.category == "delivery_issue" else "in_progress")
    return self._build_action_proposal(
        action_type=action_type,
        ticket=ticket,
        triage=triage,
        order=order,
        rationale="test stub standard-resolution path",
        confidence=0.82,
        target_status=target_status,
        requires_approval=False,
        route_family=sufficiency_result.route_family,
        sufficiency_passed=sufficiency_result.sufficient,
        supporting_doc_ids=sufficiency_result.supporting_doc_ids,
        missing_sources=sufficiency_result.missing_sources,
        decision_source="llm_minimind",
    )


def _stub_draft_reply(
    self: ResolutionAgent,
    ticket: TicketRecord,
    triage: TriageResult,
    proposed_action: ActionProposal,
    execution_result,
    retrieved_docs,
    templates,
    review_decision=None,
) -> TicketResponse:
    del self, triage, templates, review_decision
    if proposed_action.action_type == "refund":
        customer_reply = "我们已提交退款审核，财务会尽快同步后续进展。"
    elif proposed_action.action_type == "escalation":
        customer_reply = "当前问题已升级给相关团队优先处理，我们会持续同步进展。"
    elif proposed_action.action_type == "request_info":
        customer_reply = "为便于继续处理，请补充订单号、截图或相关信息。"
    elif proposed_action.action_type == "troubleshoot":
        customer_reply = "我们正在排查相关报错与日志，请稍候。"
    else:
        customer_reply = "我们正在核验并持续跟进处理进展。"
    note = f"stub reply for {ticket.ticket_id}"
    return TicketResponse(
        customer_reply=customer_reply,
        internal_note=note,
        status=proposed_action.target_status,
        citations=[doc.citations[0] for doc in retrieved_docs[:1] if doc.citations],
        draft_source="llm_minimind",
    )


def _stub_fact_check_reply(
    self: ResolutionAgent,
    ticket: TicketRecord,
    triage: TriageResult,
    proposed_action: ActionProposal,
    execution_result,
    retrieved_docs,
    response: TicketResponse,
):
    del self, ticket, triage, proposed_action, execution_result
    supported_claim_ratio = 1.0 if retrieved_docs or response.status == "waiting_on_customer" else 0.8
    fact_check = ReplyFactCheckResult(
        grounded=True,
        policy_safe=True,
        hallucination_risk=False,
        supported_claim_ratio=supported_claim_ratio,
        unsupported_claims=[],
        rewrite_needed=False,
        review_note="stub fact check passed",
        passed=True,
        rewritten=False,
        downgraded=False,
    )
    response.review_passed = True
    response.review_grounded = True
    response.review_policy_safe = True
    response.review_hallucination_risk = False
    response.review_supported_claim_ratio = supported_claim_ratio
    response.review_note = fact_check.review_note
    return response, fact_check


@pytest.fixture(autouse=True)
def stub_llm_agents(monkeypatch):
    monkeypatch.setattr(TriageAgent, "analyze", _stub_analyze)
    monkeypatch.setattr(ResolutionAgent, "propose_action", _stub_propose_action)
    monkeypatch.setattr(ResolutionAgent, "draft_reply", _stub_draft_reply)
    monkeypatch.setattr(ResolutionAgent, "fact_check_reply", _stub_fact_check_reply)
