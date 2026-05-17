from __future__ import annotations

from pathlib import Path

import pytest

from ticketflow.conversation_agent import AgentChatContext, handle_agent_chat
from ticketflow.graph import TicketFlowRunner
from ticketflow.skills import SkillRegistry, SkillRuntime, load_skill_manifest
from ticketflow.worker import TASK_REGISTRY, run_skill_now


def _write_skill(root: Path, *, skill_id: str = "sample-skill", enabled: bool = True) -> Path:
    skill_dir = root / skill_id
    skill_dir.mkdir(parents=True)
    (skill_dir / "skill.yaml").write_text(
        "\n".join(
            [
                f"skill_id: {skill_id}",
                "name: Sample Skill",
                "version: 1.0.0",
                "description: Read-only sample skill.",
                "risk_level: low",
                f"enabled: {'true' if enabled else 'false'}",
                "approval_required: false",
                "entrypoint: ticketflow.skills:noop",
                "permissions:",
                "  - tickets:read",
                "tools:",
                "  - get_ticket",
            ]
        ),
        encoding="utf-8",
    )
    (skill_dir / "SKILL.md").write_text("# Sample Skill\n\nRead-only sample.", encoding="utf-8")
    return skill_dir


def test_skill_manifest_loader_validates_required_fields(tmp_path):
    skill_dir = _write_skill(tmp_path)

    manifest = load_skill_manifest(skill_dir)

    assert manifest.skill_id == "sample-skill"
    assert manifest.version == "1.0.0"
    assert manifest.permissions == ["tickets:read"]

    broken_dir = tmp_path / "broken-skill"
    broken_dir.mkdir()
    (broken_dir / "skill.yaml").write_text("skill_id: broken-skill\nrisk_level: low\n", encoding="utf-8")

    with pytest.raises(ValueError, match="version"):
        load_skill_manifest(broken_dir)


def test_skill_registry_reload_is_idempotent_and_disable_blocks_runtime(ticketflow_project, tmp_path):
    _write_skill(tmp_path)
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    runner.reset_demo_data()
    registry = SkillRegistry(runner.repository, tmp_path)

    first = registry.reload()
    second = registry.reload()
    skills = [skill for skill in runner.repository.list_agent_skills() if skill["skill_id"] == "sample-skill"]
    runner.repository.set_agent_skill_enabled("sample-skill", False)
    runtime = SkillRuntime(runner.repository, project_root=ticketflow_project)
    result = runtime.run_skill("sample-skill", input_payload={}, actor="tester")
    runs = runner.repository.list_skill_runs(skill_id="sample-skill")
    runner.close()

    assert first["loaded_count"] == 1
    assert second["loaded_count"] == 1
    assert len(skills) == 1
    assert result.status == "rejected"
    assert result.error_message == "skill_disabled"
    assert runs[0]["status"] == "rejected"


def test_builtin_ticket_ops_skill_queries_ticket_and_records_run(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    runner.reset_demo_data()
    registry = SkillRegistry(runner.repository, Path.cwd() / "skills")
    registry.reload()
    ticket = runner.list_open_tickets(limit=1)[0]
    runtime = SkillRuntime(runner.repository, project_root=ticketflow_project)

    result = runtime.run_skill(
        "ticketflow-ops",
        input_payload={"operation": "query_ticket", "ticket_id": ticket.ticket_id},
        actor="tester",
        ticket_id=ticket.ticket_id,
    )
    runs = runner.repository.list_skill_runs(skill_id="ticketflow-ops")
    runner.close()

    assert result.status == "succeeded"
    assert result.result["ticket"]["ticket_id"] == ticket.ticket_id
    assert result.requires_approval is False
    assert runs[0]["skill_id"] == "ticketflow-ops"
    assert runs[0]["status"] == "succeeded"


def test_agent_routes_evidence_explanation_through_skill_runtime(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(
        ticketflow_project,
        overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory"},
    )
    runner.reset_demo_data()
    SkillRegistry(runner.repository, Path.cwd() / "skills").reload()
    ticket = runner.list_open_tickets(limit=1)[0]
    context = AgentChatContext(
        repository=runner.repository,
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "KG_BACKEND": "memory"},
    )

    result = handle_agent_chat(f"explain evidence chain for {ticket.ticket_id}", context)
    runner.close()

    assert result["intent"] == "explain_ticket_graph"
    assert result["data"]["skill_run"]["skill_id"] == "ticketflow-kg-memory"
    assert result["data"]["skill_run"]["status"] == "succeeded"


def test_worker_registry_and_run_skill_now(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    runner.reset_demo_data()
    SkillRegistry(runner.repository, Path.cwd() / "skills").reload()
    ticket = runner.list_open_tickets(limit=1)[0]
    runner.close()

    result = run_skill_now(
        skill_id="ticketflow-ops",
        input_payload={"operation": "query_ticket", "ticket_id": ticket.ticket_id},
        actor="worker-test",
        ticket_id=ticket.ticket_id,
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash"},
    )

    assert "run_skill" in TASK_REGISTRY
    assert result["status"] == "succeeded"
    assert result["result"]["ticket"]["ticket_id"] == ticket.ticket_id
