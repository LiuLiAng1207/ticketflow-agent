from __future__ import annotations

from ticketflow.graph import TicketFlowRunner
from ticketflow.worker import (
    TASK_REGISTRY,
    deliver_outbox_event_now,
    describe_registered_tasks,
    run_local_once,
    run_ticket_workflow_now,
)


def _first_ticket_id(runner: TicketFlowRunner, expected_category: str, *, require_linked_order: bool = False) -> str:
    for ticket in runner.list_open_tickets(limit=500):
        if ticket.expected_category == expected_category and (not require_linked_order or ticket.linked_order_id):
            return ticket.ticket_id
    raise AssertionError(f"No seeded ticket found for category {expected_category}")


def test_worker_registry_contains_first_wave_tasks():
    assert set(TASK_REGISTRY) == {
        "run_ticket_workflow",
        "send_outbox_email",
        "run_claw_task",
        "build_knowledge_graph",
    }


def test_worker_can_run_local_once_without_queue():
    result = run_local_once()

    assert result["mode"] == "local"
    assert result["status"] == "ok"
    assert "run_ticket_workflow" in result["registered_tasks"]


def test_describe_registered_tasks_is_json_friendly():
    tasks = describe_registered_tasks()

    assert tasks
    assert all("task_name" in task and "description" in task for task in tasks)


def test_worker_runs_ticket_workflow_and_updates_task(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    runner.reset_demo_data()
    ticket = runner.list_open_tickets(limit=1)[0]
    task = runner.repository.create_workflow_task(ticket_id=ticket.ticket_id, mode="async")
    runner.close()

    result = run_ticket_workflow_now(
        task_id=task["task_id"],
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash"},
    )

    assert result["status"] == "succeeded"
    assert result["task_id"] == task["task_id"]


def test_worker_records_failed_workflow_task(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    runner.reset_demo_data()
    task = runner.repository.create_workflow_task(ticket_id="UNKNOWN-TICKET", mode="async")
    runner.close()

    result = run_ticket_workflow_now(
        task_id=task["task_id"],
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash"},
    )

    assert result["status"] == "failed"
    assert "UNKNOWN-TICKET" in result["error_message"]


def test_worker_marks_interrupted_workflow_waiting_for_durable_approval(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(
        ticketflow_project,
        overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )
    runner.reset_demo_data()
    ticket_id = _first_ticket_id(runner, "billing_refund", require_linked_order=True)
    task = runner.repository.create_workflow_task(ticket_id=ticket_id, mode="async")
    runner.close()

    result = run_ticket_workflow_now(
        task_id=task["task_id"],
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )

    reloaded = TicketFlowRunner.from_project_root(
        ticketflow_project,
        overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )
    approvals = reloaded.repository.list_approval_requests(status="pending")
    reloaded.close()

    assert result["status"] == "waiting_approval"
    assert result["thread_id"]
    assert result["approval_id"]
    assert approvals
    assert approvals[0]["approval_id"] == result["approval_id"]
    assert approvals[0]["thread_id"] == result["thread_id"]
    assert approvals[0]["tool_name"] == "issue_refund_request"


def test_worker_delivers_outbox_email_once(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(
        ticketflow_project,
        overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )
    runner.reset_demo_data()
    ticket_id = runner.list_open_tickets(limit=1)[0].ticket_id
    event = runner.repository.create_outbox_event(
        ticket_id=ticket_id,
        operation_type="incident_email",
        business_key=f"incident:{ticket_id}",
        payload={
            "recipient": "ops@example.com",
            "subject": "[TicketFlow] incident",
            "provider_message_id": "provider-1",
        },
    )
    runner.close()

    first = deliver_outbox_event_now(
        event_id=event["event_id"],
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )
    second = deliver_outbox_event_now(
        event_id=event["event_id"],
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )

    reloaded = TicketFlowRunner.from_project_root(
        ticketflow_project,
        overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )
    deliveries = reloaded.repository.list_external_email_deliveries(ticket_id)
    reloaded.close()

    assert first["status"] == "delivered"
    assert second["status"] == "delivered"
    assert len(deliveries) == 1
    assert deliveries[0].op_type == "incident_email"
