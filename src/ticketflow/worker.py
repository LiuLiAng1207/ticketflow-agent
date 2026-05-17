from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:  # Celery is a production dependency, but tests can still import this module before install.
    from celery import Celery
except ModuleNotFoundError:  # pragma: no cover - exercised only in partially installed environments.
    Celery = None  # type: ignore[assignment]

from .graph import TicketFlowRunner
from .knowledge_graph import build_ticket_graph_from_repository, create_knowledge_graph_store
from .service_settings import ServiceSettings
from .skills import SkillRegistry, SkillRuntime
from .claw import ClawRegistry, ClawRuntime


@dataclass(frozen=True, slots=True)
class WorkerTask:
    task_name: str
    description: str
    queue: str = "default"


TASK_REGISTRY: dict[str, WorkerTask] = {
    "run_ticket_workflow": WorkerTask(
        task_name="run_ticket_workflow",
        description="Run the TicketFlow LangGraph workflow for one ticket.",
        queue="workflow",
    ),
    "send_outbox_email": WorkerTask(
        task_name="send_outbox_email",
        description="Deliver an already recorded outbox email event.",
        queue="external_ops",
    ),
    "run_claw_task": WorkerTask(
        task_name="run_claw_task",
        description="Run a Claw-style benchmark task and record its trajectory.",
        queue="evaluation",
    ),
    "build_knowledge_graph": WorkerTask(
        task_name="build_knowledge_graph",
        description="Build or refresh knowledge graph edges from ticket evidence.",
        queue="knowledge_graph",
    ),
    "run_skill": WorkerTask(
        task_name="run_skill",
        description="Run an allowlisted TicketFlow Skill through the audited Skill Runtime.",
        queue="skills",
    ),
}


def create_celery_app(settings: ServiceSettings | None = None):
    if Celery is None:  # pragma: no cover - defensive path for environments missing deps.
        raise RuntimeError("Celery is not installed. Run `python -m pip install -e .[dev]` first.")
    service_settings = settings or ServiceSettings.from_project_root()
    broker_url = service_settings.celery_broker_url or "memory://"
    result_backend = service_settings.celery_result_backend or "cache+memory://"
    app = Celery("ticketflow", broker=broker_url, backend=result_backend)
    app.conf.task_always_eager = service_settings.celery_task_always_eager
    app.conf.task_eager_propagates = False
    app.conf.task_serializer = "json"
    app.conf.result_serializer = "json"
    app.conf.accept_content = ["json"]
    return app


celery_app = create_celery_app()


def describe_registered_tasks() -> list[dict[str, str]]:
    return [
        {
            "task_name": task.task_name,
            "description": task.description,
            "queue": task.queue,
        }
        for task in TASK_REGISTRY.values()
    ]


def run_local_once(settings: ServiceSettings | None = None) -> dict[str, object]:
    service_settings = settings or ServiceSettings.from_project_root()
    return {
        "status": "ok",
        "mode": "local",
        "app_env": service_settings.app_env,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "registered_tasks": list(TASK_REGISTRY),
        "celery_task_always_eager": service_settings.celery_task_always_eager,
        "note": "Celery worker skeleton is configured; production queues are enabled when CELERY_TASK_ALWAYS_EAGER=false.",
    }


def run_ticket_workflow_now(
    *,
    task_id: str,
    project_root: str | Path,
    runner_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    runner = TicketFlowRunner.from_project_root(project_root, overrides=runner_overrides)
    try:
        task = runner.repository.get_workflow_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        runner.repository.mark_workflow_task_running(task_id)
        result = runner.run_ticket(
            str(task["ticket_id"]),
            thread_id=task.get("thread_id") or None,
            workflow_task_id=task_id,
        )
        if result.interrupted:
            approval_id = None
            if result.interrupt_payload:
                approval_id = result.interrupt_payload.get("approval_id")
            return runner.repository.wait_workflow_task_for_approval(
                task_id,
                thread_id=str(result.state["thread_id"]),
                approval_id=str(approval_id) if approval_id else None,
            )
        completed = runner.repository.complete_workflow_task(task_id, result=result.model_dump(mode="json"))
        enqueue_pending_outbox_for_ticket(
            ticket_id=str(task["ticket_id"]),
            project_root=project_root,
            runner_overrides=runner_overrides,
        )
        return completed
    except Exception as exc:
        try:
            return runner.repository.fail_workflow_task(task_id, error_message=str(exc))
        except Exception:
            raise exc
    finally:
        runner.close()


@celery_app.task(name="ticketflow.run_ticket_workflow")
def run_ticket_workflow_task(task_id: str, project_root: str, runner_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    return run_ticket_workflow_now(
        task_id=task_id,
        project_root=project_root,
        runner_overrides=runner_overrides,
    )


def enqueue_run_ticket_workflow(
    *,
    task_id: str,
    project_root: str | Path,
    runner_overrides: dict[str, str] | None = None,
) -> str:
    async_result = run_ticket_workflow_task.delay(str(task_id), str(project_root), runner_overrides)
    return str(async_result.id)


def enqueue_deliver_outbox_event(
    *,
    event_id: str,
    project_root: str | Path,
    runner_overrides: dict[str, str] | None = None,
) -> str:
    async_result = deliver_outbox_event_task.delay(str(event_id), str(project_root), runner_overrides)
    return str(async_result.id)


def enqueue_pending_outbox_for_ticket(
    *,
    ticket_id: str,
    project_root: str | Path,
    runner_overrides: dict[str, str] | None = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    runner = TicketFlowRunner.from_project_root(project_root, overrides=runner_overrides)
    try:
        pending_events = [
            event
            for event in runner.repository.list_outbox_events(status="pending", limit=limit)
            if str(event["ticket_id"]) == ticket_id
        ]
    finally:
        runner.close()

    enqueued: list[dict[str, Any]] = []
    for event in pending_events:
        celery_task_id = enqueue_deliver_outbox_event(
            event_id=str(event["event_id"]),
            project_root=project_root,
            runner_overrides=runner_overrides,
        )
        enqueued.append({**event, "celery_task_id": celery_task_id})
    return enqueued


def build_knowledge_graph_now(
    *,
    ticket_id: str,
    project_root: str | Path,
    task_id: str | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    settings = ServiceSettings.from_project_root(project_root, overrides=runner_overrides)
    runner = TicketFlowRunner.from_project_root(project_root, overrides=runner_overrides)
    store = create_knowledge_graph_store(settings)
    try:
        graph = build_ticket_graph_from_repository(runner.repository, ticket_id, task_id=task_id)
        result = store.upsert_ticket_graph(graph)
        return {
            **result,
            "ticket_id": ticket_id,
            "task_id": task_id,
            "graph": {"node_count": graph["node_count"], "edge_count": graph["edge_count"]},
        }
    finally:
        store.close()
        runner.close()


def run_claw_task_now(
    *,
    task_id: str,
    project_root: str | Path,
    actor: str = "ticketflow-worker",
    pass_k: int | None = None,
    run_id: str | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    settings = ServiceSettings.from_project_root(project_root, overrides=runner_overrides)
    runner = TicketFlowRunner.from_project_root(project_root, overrides=runner_overrides)
    try:
        ClawRegistry(runner.repository, settings.claw_tasks_dir).reload()
        SkillRegistry(runner.repository, settings.skills_dir).reload()
        result = ClawRuntime(
            runner.repository,
            project_root=project_root,
            runner_overrides=runner_overrides,
        ).run_task(task_id, actor=actor, pass_k=pass_k, run_id=run_id)
        return result.model_dump(mode="json")
    finally:
        runner.close()


@celery_app.task(name="ticketflow.run_claw_task")
def run_claw_task(
    task_id: str,
    project_root: str,
    actor: str = "ticketflow-worker",
    pass_k: int | None = None,
    run_id: str | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    return run_claw_task_now(
        task_id=task_id,
        project_root=project_root,
        actor=actor,
        pass_k=pass_k,
        run_id=run_id,
        runner_overrides=runner_overrides,
    )


def enqueue_run_claw_task(
    *,
    task_id: str,
    project_root: str | Path,
    actor: str = "ticketflow-worker",
    pass_k: int | None = None,
    run_id: str | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> str:
    async_result = run_claw_task.delay(str(task_id), str(project_root), actor, pass_k, run_id, runner_overrides)
    return str(async_result.id)


@celery_app.task(name="ticketflow.build_knowledge_graph")
def build_knowledge_graph_task(
    ticket_id: str,
    project_root: str,
    task_id: str | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    return build_knowledge_graph_now(
        ticket_id=ticket_id,
        project_root=project_root,
        task_id=task_id,
        runner_overrides=runner_overrides,
    )


def run_skill_now(
    *,
    skill_id: str,
    input_payload: dict[str, Any],
    actor: str,
    project_root: str | Path,
    ticket_id: str | None = None,
    idempotency_key: str | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    settings = ServiceSettings.from_project_root(project_root, overrides=runner_overrides)
    runner = TicketFlowRunner.from_project_root(project_root, overrides=runner_overrides)
    store = create_knowledge_graph_store(settings)
    try:
        SkillRegistry(runner.repository, settings.skills_dir).reload()
        result = SkillRuntime(
            runner.repository,
            project_root=project_root,
            runner_overrides=runner_overrides,
            knowledge_graph_store=store,
        ).run_skill(
            skill_id,
            input_payload=input_payload,
            actor=actor,
            ticket_id=ticket_id,
            idempotency_key=idempotency_key,
        )
        return result.model_dump(mode="json")
    finally:
        store.close()
        runner.close()


@celery_app.task(name="ticketflow.run_skill")
def run_skill_task(
    skill_id: str,
    input_payload: dict[str, Any],
    actor: str,
    project_root: str,
    ticket_id: str | None = None,
    idempotency_key: str | None = None,
    runner_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    return run_skill_now(
        skill_id=skill_id,
        input_payload=input_payload,
        actor=actor,
        ticket_id=ticket_id,
        idempotency_key=idempotency_key,
        project_root=project_root,
        runner_overrides=runner_overrides,
    )


def deliver_outbox_event_now(
    *,
    event_id: str,
    project_root: str | Path,
    runner_overrides: dict[str, str] | None = None,
) -> dict[str, Any]:
    runner = TicketFlowRunner.from_project_root(project_root, overrides=runner_overrides)
    try:
        event = runner.repository.get_outbox_event(event_id)
        if event is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        if event["status"] == "delivered":
            return event

        lock = runner.repository.acquire_external_operation_lock(
            lock_key=f"outbox:{event_id}",
            ticket_id=str(event["ticket_id"]),
            operation_type=str(event["operation_type"]),
            acquired_by="ticketflow-worker",
        )
        if not lock["acquired"]:
            return event

        payload = event["payload"] if isinstance(event["payload"], dict) else {}
        if payload.get("simulate_error"):
            raise RuntimeError(str(payload["simulate_error"]))

        operation_type = str(event["operation_type"])
        if operation_type in {"incident_email", "kb_candidate_email"}:
            recipient = str(payload.get("recipient") or (payload.get("recipients") or ["-"])[0])
            subject = str(payload.get("subject") or f"[TicketFlow] {operation_type} {event['ticket_id']}")
            runner.repository.create_external_email_delivery(
                ticket_id=str(event["ticket_id"]),
                message_type=operation_type,
                recipient=recipient,
                subject=subject,
                status="sent",
                provider_message_id=str(payload.get("provider_message_id") or event_id),
                latency_ms=0,
                error_message=None,
                payload=payload,
            )
        elif operation_type == "issue_refund_request":
            args = payload.get("tool_args") if isinstance(payload.get("tool_args"), dict) else payload
            runner.tools.issue_refund_request(**args)
            if payload.get("target_status"):
                runner.tools.update_ticket_status(str(event["ticket_id"]), str(payload["target_status"]))
        elif operation_type == "create_escalation":
            args = payload.get("tool_args") if isinstance(payload.get("tool_args"), dict) else payload
            runner.tools.create_escalation(**args)
            if payload.get("target_status"):
                runner.tools.update_ticket_status(str(event["ticket_id"]), str(payload["target_status"]))

        return runner.repository.mark_outbox_event_delivered(event_id)
    except Exception as exc:
        try:
            return runner.repository.mark_outbox_event_failed(event_id, str(exc))
        except Exception:
            raise exc
    finally:
        runner.close()


@celery_app.task(name="ticketflow.deliver_outbox_event")
def deliver_outbox_event_task(event_id: str, project_root: str, runner_overrides: dict[str, str] | None = None) -> dict[str, Any]:
    return deliver_outbox_event_now(
        event_id=event_id,
        project_root=project_root,
        runner_overrides=runner_overrides,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TicketFlow worker skeleton")
    parser.add_argument("--mode", choices=["local"], default="local", help="Worker mode for phase 1.")
    parser.add_argument("--once", action="store_true", help="Start, print health information, and exit.")
    parser.add_argument("--list-tasks", action="store_true", help="Print registered worker tasks and exit.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_tasks:
        print(json.dumps({"tasks": describe_registered_tasks()}, ensure_ascii=False, indent=2))
        return

    if args.mode == "local":
        payload = run_local_once()
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported worker mode: {args.mode}")
