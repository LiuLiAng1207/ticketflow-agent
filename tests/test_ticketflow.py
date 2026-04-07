from __future__ import annotations

from ticketflow.evals import run_evaluation
from ticketflow.models import ReviewDecision


class _FakeEmailClient:
    def call_tool(self, tool_name: str, payload: dict[str, object]) -> dict[str, object]:
        return {
            "status": "sent",
            "delivery_id": f"delivery-{tool_name}",
            "provider_message_id": f"message-{tool_name}",
            "sent_at": "2026-03-31T00:00:00",
            "recipients": payload["recipients"],
        }


class _BrokenEmailClient:
    def call_tool(self, tool_name: str, payload: dict[str, object]) -> dict[str, object]:
        raise RuntimeError(f"{tool_name} failed")


def _find_ticket(runner, category: str, *, missing_order: bool | None = None, tier: str | None = None, refundable: bool | None = None):
    for ticket in runner.list_open_tickets(limit=300):
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
    raise AssertionError(f"Unable to find ticket for category={category}")


def test_refund_ticket_interrupts_before_execution(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=False, refundable=True)
    result = runner.run_ticket(ticket.ticket_id, thread_id="refund-test")

    assert result.interrupted is True
    assert result.interrupt_payload["proposed_action"]["action_type"] == "refund"
    assert result.state["approval_state"] == "pending_review"
    assert result.state["triage_result"].category == "billing_refund"


def test_resume_after_approval_executes_refund(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=False, refundable=True)
    initial = runner.run_ticket(ticket.ticket_id, thread_id="refund-approve")
    resumed = runner.resume_ticket(initial.state["thread_id"], ReviewDecision(decision="approve", comment="Proceed"))

    assert resumed.interrupted is False
    assert resumed.state["execution_result"]["status"] == "executed"
    assert resumed.state["draft_reply"].status == "pending_finance"


def test_missing_order_refund_requests_more_info(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=True)
    result = runner.run_ticket(ticket.ticket_id, thread_id="refund-missing")

    assert result.interrupted is False
    assert result.state["proposed_action"].action_type == "request_info"
    assert result.state["draft_reply"].status == "waiting_on_customer"


def test_enterprise_outage_triggers_escalation_and_interrupt(runner):
    ticket = _find_ticket(runner, "technical_issue", tier="enterprise")
    result = runner.run_ticket(ticket.ticket_id, thread_id="enterprise-outage")

    assert result.interrupted is True
    assert result.state["proposed_action"].action_type == "escalation"
    assert result.state["triage_result"].priority in {"high", "urgent"}


def test_thread_snapshot_preserves_pending_state(runner):
    ticket = _find_ticket(runner, "delivery_issue")
    result = runner.run_ticket(ticket.ticket_id, thread_id="snapshot-thread")

    if result.interrupted:
        snapshot = runner.get_snapshot("snapshot-thread")
        assert snapshot["next"] == ["approval_gate"]
        assert snapshot["values"]["ticket"].ticket_id == ticket.ticket_id


def test_rag_retrieve_context_contains_policy_kb_and_history_hits(runner):
    ticket = _find_ticket(runner, "billing_refund", missing_order=False)
    result = runner.run_ticket(ticket.ticket_id, thread_id="rag-coverage")

    stats = result.state["retrieval_stats"]
    assert stats.source_counts.get("kb", 0) >= 1
    assert stats.source_counts.get("policy", 0) >= 1
    assert stats.source_counts.get("history", 0) >= 1


def test_seed_tickets_keep_public_corpus_provenance(runner):
    ticket = runner.list_open_tickets(limit=1)[0]

    assert ticket.source_dataset == "Tobi-Bueck/customer-support-tickets"
    assert ticket.source_ticket_ref
    assert ticket.source_queue
    assert ticket.source_subject


def test_escalation_dispatches_external_emails_when_mcp_is_available(runner_factory):
    runner = runner_factory(
        MODEL_BACKEND="rule",
        INCIDENT_EMAIL_TO="ops@example.com",
        KB_OPS_EMAIL_TO="kb@example.com",
    )
    runner.email_client = _FakeEmailClient()

    ticket = _find_ticket(runner, "technical_issue", tier="enterprise")
    initial = runner.run_ticket(ticket.ticket_id, thread_id="mcp-success")
    resumed = runner.resume_ticket(initial.state["thread_id"], ReviewDecision(decision="approve", comment="ok"))

    external_ops = resumed.state["external_ops"]
    assert any(op.op_type == "incident_email" and op.status == "sent" for op in external_ops)
    assert any(op.op_type == "kb_candidate_email" and op.status == "sent" for op in external_ops)

    deliveries = runner.repository.list_external_email_deliveries(ticket.ticket_id)
    assert {record.op_type for record in deliveries} >= {"incident_email", "kb_candidate_email"}


def test_external_email_failure_is_non_blocking(runner_factory):
    runner = runner_factory(
        MODEL_BACKEND="rule",
        INCIDENT_EMAIL_TO="ops@example.com",
        KB_OPS_EMAIL_TO="kb@example.com",
    )
    runner.email_client = _BrokenEmailClient()

    ticket = _find_ticket(runner, "technical_issue", tier="enterprise")
    initial = runner.run_ticket(ticket.ticket_id, thread_id="mcp-failure")
    resumed = runner.resume_ticket(initial.state["thread_id"], ReviewDecision(decision="approve", comment="ok"))

    assert resumed.state["draft_reply"].status == "escalated"
    assert any(op.status == "failed" for op in resumed.state["external_ops"])


def test_evaluation_report_contains_granular_node_metrics(ticketflow_project):
    report = run_evaluation(ticketflow_project, sample_size=8, overrides={"MODEL_BACKEND": "rule"})

    for key in (
        "category_macro_f1",
        "sla_risk_recall",
        "action_type_accuracy",
        "approval_required_recall",
        "target_status_accuracy",
        "groundedness_rate",
        "policy_violation_rate",
        "hallucination_rate",
        "reply_rubric_proxy_mean",
    ):
        assert key in report
