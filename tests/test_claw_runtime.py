from __future__ import annotations

from pathlib import Path

import pytest

from ticketflow.claw import ClawRegistry, ClawRuntime, load_claw_task_manifest
from ticketflow.skills import SkillRegistry
from ticketflow.worker import TASK_REGISTRY, run_claw_task_now


def _write_task(path: Path, *, task_id: str, ticket_id: str = "TCK-0001") -> None:
    path.write_text(
        "\n".join(
            [
                f"task_id: {task_id}",
                "name: Query ticket evidence",
                "goal: Query a ticket through an allowlisted Skill and verify a grounded response.",
                "category: ticket_ops",
                "pass_k: 3",
                "input:",
                "  operation: skill",
                "  skill_id: ticketflow-ops",
                "  skill_input:",
                "    operation: query_ticket",
                f"    ticket_id: {ticket_id}",
                "expected:",
                f"  ticket_id: {ticket_id}",
                "  required_result_keys:",
                "    - ticket",
                "allowed_tools:",
                "  - skill:ticketflow-ops",
                "forbidden_tools:",
                "  - shell",
                "verifier:",
                "  type: deterministic",
            ]
        ),
        encoding="utf-8",
    )


def test_claw_manifest_validation_requires_goal_and_verifier(tmp_path):
    task_path = tmp_path / "sample-task.yaml"
    _write_task(task_path, task_id="sample-task")

    manifest = load_claw_task_manifest(task_path)

    assert manifest.task_id == "sample-task"
    assert manifest.pass_k == 3
    assert manifest.verifier["type"] == "deterministic"

    invalid_path = tmp_path / "invalid.yaml"
    invalid_path.write_text("task_id: invalid\nname: Invalid\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_claw_task_manifest(invalid_path)


def test_claw_registry_reload_is_idempotent(runner, tmp_path):
    _write_task(tmp_path / "sample-task.yaml", task_id="sample-task")
    registry = ClawRegistry(runner.repository, tmp_path)

    first = registry.reload()
    second = registry.reload()
    tasks = runner.repository.list_claw_tasks()

    assert first["loaded_count"] == 1
    assert second["loaded_count"] == 1
    assert [task for task in tasks if task["task_id"] == "sample-task"]
    assert len([task for task in tasks if task["task_id"] == "sample-task"]) == 1


def test_claw_runtime_runs_pass3_and_records_scores(runner, tmp_path, ticketflow_project):
    ticket = runner.list_open_tickets(limit=1)[0]
    _write_task(tmp_path / "sample-task.yaml", task_id="sample-task", ticket_id=ticket.ticket_id)
    ClawRegistry(runner.repository, tmp_path).reload()
    SkillRegistry(runner.repository, Path.cwd() / "skills").reload()

    result = ClawRuntime(
        runner.repository,
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory", "CELERY_TASK_ALWAYS_EAGER": "true"},
    ).run_task("sample-task", actor="runtime-test", pass_k=3)
    stored = runner.repository.get_claw_run(result.run_id)
    leaderboard = runner.repository.list_claw_leaderboard()

    assert result.status == "succeeded"
    assert len(result.attempts) == 3
    assert all(attempt["status"] == "succeeded" for attempt in result.attempts)
    assert all(attempt["score"]["passed"] for attempt in result.attempts)
    assert stored is not None
    assert len(stored["attempts"]) == 3
    assert stored["summary"]["pass_count"] == 3
    assert any(row["task_id"] == "sample-task" for row in leaderboard)


def test_worker_registry_and_run_claw_task_now(ticketflow_project):
    assert "run_claw_task" in TASK_REGISTRY

    result = run_claw_task_now(
        task_id="claw-query-ticket-evidence",
        project_root=ticketflow_project,
        actor="worker-test",
        pass_k=1,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory", "CELERY_TASK_ALWAYS_EAGER": "true"},
    )

    assert result["task_id"] == "claw-query-ticket-evidence"
    assert result["status"] == "succeeded"
    assert result["summary"]["attempt_count"] == 1
