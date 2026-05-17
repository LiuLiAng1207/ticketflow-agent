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
from .service_settings import ServiceSettings


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
        result = runner.run_ticket(str(task["ticket_id"]), thread_id=task.get("thread_id") or None)
        completed = runner.repository.complete_workflow_task(task_id, result=result.model_dump(mode="json"))
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
