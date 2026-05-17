from __future__ import annotations

from ticketflow.graph import TicketFlowRunner
from ticketflow.worker import TASK_REGISTRY, describe_registered_tasks, run_local_once, run_ticket_workflow_now


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
