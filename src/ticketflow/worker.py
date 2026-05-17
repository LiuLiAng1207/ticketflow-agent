from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone

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
        "note": "Celery is not enabled in phase 1; this worker only exposes the stable task registry.",
    }


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
