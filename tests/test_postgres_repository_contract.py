from __future__ import annotations

import os
from pathlib import Path

import pytest

from ticketflow.postgres_repository import POSTGRES_SCHEMA, PostgresTicketFlowRepository


REQUIRED_TABLES = {
    "customers",
    "orders",
    "tickets",
    "kb_articles",
    "policies",
    "reply_templates",
    "ticket_history",
    "ticket_attachments",
    "attachment_evidence",
    "escalations",
    "refund_requests",
    "audit_log",
    "external_email_deliveries",
    "workflow_tasks",
    "outbox_events",
    "idempotency_keys",
    "external_operation_locks",
    "approval_requests",
    "approval_decisions",
}

REQUIRED_METHODS = {
    "bootstrap",
    "reset_from_seed",
    "load_seed_directory",
    "list_open_tickets",
    "get_ticket",
    "get_customer_profile",
    "get_order_status",
    "get_ticket_history",
    "list_ticket_attachments",
    "replace_attachment_evidence",
    "list_attachment_evidence",
    "list_kb_articles",
    "list_policies",
    "list_ticket_history_entries",
    "search_kb",
    "search_policy_text",
    "search_related_history",
    "lookup_policy",
    "get_reply_templates",
    "create_escalation",
    "issue_refund_request",
    "update_ticket_status",
    "save_audit_log",
    "list_audit_log",
    "create_workflow_task",
    "get_workflow_task",
    "list_workflow_tasks",
    "cancel_workflow_task",
    "record_idempotency_key",
    "create_outbox_event",
    "list_outbox_events",
    "create_approval_request",
    "get_approval_request",
    "list_approval_requests",
    "record_approval_decision",
    "create_external_email_delivery",
    "list_external_email_deliveries",
}


def test_postgres_schema_contains_business_and_control_tables():
    schema = POSTGRES_SCHEMA.lower()

    missing = [table for table in sorted(REQUIRED_TABLES) if f"create table if not exists {table}" not in schema]

    assert missing == []


def test_postgres_repository_exposes_sqlite_parity_methods():
    missing = [method for method in sorted(REQUIRED_METHODS) if not hasattr(PostgresTicketFlowRepository, method)]

    assert missing == []


@pytest.mark.skipif(
    not os.getenv("TEST_POSTGRES_DATABASE_URL"),
    reason="set TEST_POSTGRES_DATABASE_URL to run live PostgreSQL repository parity checks",
)
def test_live_postgres_repository_can_run_core_ticketflow_reads_and_controls(ticketflow_project: Path):
    repository = PostgresTicketFlowRepository(os.environ["TEST_POSTGRES_DATABASE_URL"])
    seed_dir = ticketflow_project / "data" / "seed"

    repository.reset_from_seed(seed_dir)
    ticket = repository.list_open_tickets(limit=1)[0]
    customer = repository.get_customer_profile(ticket.customer_id)
    order = repository.get_order_status(ticket.linked_order_id)
    kb_hits = repository.search_kb(ticket.body, ticket.product, limit=3)
    policy_hits = repository.lookup_policy("refund", ticket.body, limit=3)
    history_hits = repository.search_related_history(ticket.body, limit=3)

    task = repository.create_workflow_task(ticket.ticket_id, mode="async")
    outbox = repository.create_outbox_event(
        ticket_id=ticket.ticket_id,
        operation_type="incident_email",
        business_key=f"incident:{ticket.ticket_id}",
        payload={"recipient": "ops@example.com"},
    )
    approval = repository.create_approval_request(
        ticket_id=ticket.ticket_id,
        thread_id="thread-live-postgres",
        tool_name="issue_refund_request",
        tool_args={"order_id": ticket.linked_order_id},
        payload={"route_family": "refund_candidate"},
        requested_by="test",
    )
    decision = repository.record_approval_decision(
        approval_id=approval["approval_id"],
        decision="approve",
        reviewer="lead",
    )

    assert customer is not None
    assert kb_hits
    assert policy_hits
    assert history_hits
    assert order is None or order.customer_id == ticket.customer_id
    assert task["status"] == "queued"
    assert outbox["deduplicated"] is False
    assert repository.get_approval_request(approval["approval_id"])["status"] == "approved"
    assert decision["approval_id"] == approval["approval_id"]
