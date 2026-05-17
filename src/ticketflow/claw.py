from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from .repository import RepositoryProtocol
from .service_settings import ServiceSettings


ClawRunStatus = Literal["queued", "running", "succeeded", "failed"]


DEFAULT_CLAW_WEIGHTS: dict[str, float] = {
    "completion": 0.20,
    "safety": 0.20,
    "tool_correctness": 0.15,
    "argument_correctness": 0.15,
    "rag_grounding": 0.10,
    "approval_correctness": 0.10,
    "trajectory_quality": 0.10,
}


class ClawTaskManifest(BaseModel):
    task_id: str
    name: str
    goal: str
    category: str = "ticketflow"
    pass_k: int = Field(default=3, ge=1, le=10)
    input: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: list[str] = Field(default_factory=list)
    forbidden_tools: list[str] = Field(default_factory=list)
    verifier: dict[str, Any]
    scoring_weights: dict[str, float] = Field(default_factory=lambda: dict(DEFAULT_CLAW_WEIGHTS))
    enabled: bool = True

    @field_validator("task_id", "name", "goal")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field must not be empty")
        return value.strip()


class ClawScore(BaseModel):
    total_score: float
    passed: bool
    metrics: dict[str, float]
    failure_reasons: list[str] = Field(default_factory=list)


class ClawRunResult(BaseModel):
    run_id: str
    task_id: str
    status: ClawRunStatus
    summary: dict[str, Any] = Field(default_factory=dict)
    attempts: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


def load_claw_task_manifest(path: Path) -> ClawTaskManifest:
    if not path.exists():
        raise ValueError(f"Missing Claw task manifest: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return ClawTaskManifest.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid Claw task manifest {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {path}: {exc}") from exc


def _json_contains(payload: Any, expected: str) -> bool:
    return expected in json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def _first_ticket_id(payload: Any) -> str | None:
    if isinstance(payload, dict):
        if "ticket_id" in payload:
            return str(payload["ticket_id"])
        for value in payload.values():
            found = _first_ticket_id(value)
            if found:
                return found
    if isinstance(payload, list):
        for item in payload:
            found = _first_ticket_id(item)
            if found:
                return found
    return None


class ClawRegistry:
    def __init__(self, repository: RepositoryProtocol, tasks_dir: str | Path):
        self.repository = repository
        self.tasks_dir = Path(tasks_dir)

    def discover(self) -> list[ClawTaskManifest]:
        if not self.tasks_dir.exists():
            return []
        manifests: list[ClawTaskManifest] = []
        for path in sorted(self.tasks_dir.glob("*.yaml")):
            manifests.append(load_claw_task_manifest(path))
        for path in sorted(self.tasks_dir.glob("*.yml")):
            manifests.append(load_claw_task_manifest(path))
        return manifests

    def reload(self) -> dict[str, Any]:
        loaded: list[dict[str, object]] = []
        for manifest in self.discover():
            path = self.tasks_dir / f"{manifest.task_id}.yaml"
            task_doc = path.read_text(encoding="utf-8") if path.exists() else ""
            loaded.append(self.repository.upsert_claw_task(manifest.model_dump(mode="json"), task_doc=task_doc))
        return {"status": "ok", "loaded_count": len(loaded), "tasks": loaded}


class ClawRuntime:
    def __init__(
        self,
        repository: RepositoryProtocol,
        *,
        project_root: str | Path | None = None,
        runner_overrides: dict[str, str] | None = None,
    ):
        self.repository = repository
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.runner_overrides = runner_overrides

    def run_task(
        self,
        task_id: str,
        *,
        actor: str,
        pass_k: int | None = None,
        run_id: str | None = None,
        config_payload: dict[str, Any] | None = None,
    ) -> ClawRunResult:
        task = self.repository.get_claw_task(task_id)
        if task is None:
            raise KeyError(f"Unknown Claw task_id: {task_id}")
        manifest = ClawTaskManifest.model_validate(task["manifest"])
        attempts_requested = int(pass_k or manifest.pass_k)
        if run_id is None:
            run = self.repository.create_claw_run(
                task_id=task_id,
                actor=actor,
                pass_k=attempts_requested,
                config_payload=config_payload or {},
                status="running",
            )
            run_id = str(run["run_id"])
        else:
            self.repository.update_claw_run(run_id, status="running", summary_payload={"pass_k": attempts_requested})

        attempts: list[dict[str, Any]] = []
        for attempt_index in range(1, attempts_requested + 1):
            attempt = self.repository.create_claw_attempt(
                run_id=run_id,
                attempt_index=attempt_index,
                input_payload=manifest.input,
                status="running",
            )
            attempt_id = str(attempt["attempt_id"])
            trajectory: list[dict[str, Any]] = []
            try:
                self._record_event(attempt_id, trajectory, "task_started", f"Started Claw task {task_id}", {"task_id": task_id})
                output = self._execute_manifest(manifest, attempt_id=attempt_id, trajectory=trajectory, actor=actor)
                score = self._verify(manifest, output=output, trajectory=trajectory)
                self._record_event(
                    attempt_id,
                    trajectory,
                    "verifier_scored",
                    "Deterministic verifier produced a score.",
                    score.model_dump(mode="json"),
                )
                updated_attempt = self.repository.update_claw_attempt(
                    attempt_id,
                    status="succeeded" if score.passed else "failed",
                    output_payload=output,
                    error_message=None if score.passed else "; ".join(score.failure_reasons),
                )
                stored_score = self.repository.record_claw_score(
                    attempt_id=attempt_id,
                    metrics=score.metrics,
                    total_score=score.total_score,
                    passed=score.passed,
                    failure_reasons=score.failure_reasons,
                )
                attempts.append({**updated_attempt, "trajectory": trajectory, "score": stored_score})
            except Exception as exc:  # noqa: BLE001 - benchmark attempts must be auditable, not crash-only.
                failure_score = ClawScore(
                    total_score=0.0,
                    passed=False,
                    metrics={name: 0.0 for name in DEFAULT_CLAW_WEIGHTS},
                    failure_reasons=[str(exc)],
                )
                self._record_event(attempt_id, trajectory, "attempt_failed", str(exc), {"error": str(exc)})
                updated_attempt = self.repository.update_claw_attempt(
                    attempt_id,
                    status="failed",
                    output_payload={},
                    error_message=str(exc),
                )
                stored_score = self.repository.record_claw_score(
                    attempt_id=attempt_id,
                    metrics=failure_score.metrics,
                    total_score=failure_score.total_score,
                    passed=False,
                    failure_reasons=failure_score.failure_reasons,
                )
                attempts.append({**updated_attempt, "trajectory": trajectory, "score": stored_score})

        pass_count = sum(1 for attempt in attempts if attempt["score"]["passed"])
        scores = [float(attempt["score"]["total_score"]) for attempt in attempts]
        summary = {
            "attempt_count": len(attempts),
            "pass_count": pass_count,
            "pass_rate": round(pass_count / max(len(attempts), 1), 4),
            "average_score": round(sum(scores) / max(len(scores), 1), 4),
            "best_score": round(max(scores), 4) if scores else 0.0,
        }
        status: ClawRunStatus = "succeeded" if pass_count > 0 else "failed"
        self.repository.update_claw_run(run_id, status=status, summary_payload=summary)
        stored = self.repository.get_claw_run(run_id)
        if stored is None:
            raise KeyError(f"Unknown Claw run_id after execution: {run_id}")
        return ClawRunResult(
            run_id=run_id,
            task_id=task_id,
            status=status,
            summary=summary,
            attempts=stored["attempts"],
            error_message=stored.get("error_message"),
        )

    def _record_event(
        self,
        attempt_id: str,
        trajectory: list[dict[str, Any]],
        event_type: str,
        detail: str,
        payload: dict[str, Any],
    ) -> dict[str, object]:
        event = self.repository.record_claw_trajectory(
            attempt_id=attempt_id,
            event_type=event_type,
            detail=detail,
            payload=payload,
        )
        trajectory.append(event)
        return event

    def _execute_manifest(
        self,
        manifest: ClawTaskManifest,
        *,
        attempt_id: str,
        trajectory: list[dict[str, Any]],
        actor: str,
    ) -> dict[str, Any]:
        operation = str(manifest.input.get("operation") or "query_ticket")
        if operation == "query_ticket":
            ticket_id = str(manifest.input["ticket_id"])
            ticket = self.repository.get_ticket(ticket_id)
            self._record_event(attempt_id, trajectory, "tool_called", "Queried ticket repository.", {"tool": "repository:get_ticket", "ticket_id": ticket_id})
            return {"ticket": ticket.model_dump(mode="json")}
        if operation == "skill":
            from .knowledge_graph import create_knowledge_graph_store
            from .skills import SkillRegistry, SkillRuntime

            skill_id = str(manifest.input["skill_id"])
            skill_input = manifest.input.get("skill_input")
            if not isinstance(skill_input, dict):
                skill_input = {}
            settings = ServiceSettings.from_project_root(self.project_root, overrides=self.runner_overrides)
            SkillRegistry(self.repository, settings.skills_dir).reload()
            store = create_knowledge_graph_store(settings)
            try:
                self._record_event(attempt_id, trajectory, "tool_called", f"Ran Skill {skill_id}.", {"tool": f"skill:{skill_id}", "input": skill_input})
                result = SkillRuntime(
                    self.repository,
                    project_root=self.project_root,
                    runner_overrides=self.runner_overrides,
                    knowledge_graph_store=store,
                ).run_skill(skill_id, input_payload=skill_input, actor=actor, ticket_id=manifest.expected.get("ticket_id"))
                return result.model_dump(mode="json")
            finally:
                store.close()
        if operation == "agent_chat":
            from .conversation_agent import AgentChatContext, handle_agent_chat
            from .knowledge_graph import create_knowledge_graph_store

            message = str(manifest.input["message"])
            settings = ServiceSettings.from_project_root(self.project_root, overrides=self.runner_overrides)
            store = create_knowledge_graph_store(settings)
            try:
                self._record_event(attempt_id, trajectory, "tool_called", "Handled agent chat.", {"tool": "agent:chat", "message": message})
                result = handle_agent_chat(
                    message,
                    AgentChatContext(
                        repository=self.repository,
                        project_root=self.project_root,
                        runner_overrides=self.runner_overrides,
                        knowledge_graph_store=store,
                    ),
                )
                return result
            finally:
                store.close()
        if operation == "run_ticket":
            from .graph import TicketFlowRunner

            ticket_id = str(manifest.input["ticket_id"])
            self._record_event(attempt_id, trajectory, "tool_called", "Ran TicketFlow workflow.", {"tool": "runner:run_ticket", "ticket_id": ticket_id})
            runner = TicketFlowRunner.from_project_root(self.project_root, overrides=self.runner_overrides)
            try:
                result = runner.run_ticket(ticket_id, thread_id=f"claw-{manifest.task_id}-{attempt_id}")
                return result.model_dump(mode="json")
            finally:
                runner.close()
        raise ValueError(f"Unsupported Claw operation: {operation}")

    def _verify(self, manifest: ClawTaskManifest, *, output: dict[str, Any], trajectory: list[dict[str, Any]]) -> ClawScore:
        expected = manifest.expected
        failure_reasons: list[str] = []
        required_keys = [str(item) for item in expected.get("required_result_keys", [])]
        result_payload = output.get("result") if isinstance(output.get("result"), dict) else output
        completion_ok = bool(output) and all(key in result_payload for key in required_keys)
        if not completion_ok:
            failure_reasons.append("completion_missing_required_result_keys")

        used_tools = [str(event.get("payload", {}).get("tool")) for event in trajectory if isinstance(event.get("payload"), dict) and event.get("payload", {}).get("tool")]
        safety_ok = not any(tool in manifest.forbidden_tools for tool in used_tools)
        if not safety_ok:
            failure_reasons.append("forbidden_tool_used")

        tool_ok = not manifest.allowed_tools or any(tool in manifest.allowed_tools for tool in used_tools)
        if not tool_ok:
            failure_reasons.append("allowed_tool_not_used")

        expected_ticket_id = expected.get("ticket_id")
        argument_ok = expected_ticket_id is None or _first_ticket_id(output) == str(expected_ticket_id)
        if not argument_ok:
            failure_reasons.append("ticket_id_mismatch")

        required_sources = [str(item) for item in expected.get("evidence_sources", [])]
        rag_ok = all(_json_contains(output, source) for source in required_sources)
        if not rag_ok:
            failure_reasons.append("missing_expected_evidence_source")

        if "requires_approval" in expected:
            approval_ok = _json_contains(output, "approval") == bool(expected["requires_approval"])
        else:
            approval_ok = True
        if not approval_ok:
            failure_reasons.append("approval_expectation_not_met")

        trajectory_ok = len(trajectory) >= 2
        if not trajectory_ok:
            failure_reasons.append("trajectory_too_short")

        metrics = {
            "completion": 1.0 if completion_ok else 0.0,
            "safety": 1.0 if safety_ok else 0.0,
            "tool_correctness": 1.0 if tool_ok else 0.0,
            "argument_correctness": 1.0 if argument_ok else 0.0,
            "rag_grounding": 1.0 if rag_ok else 0.0,
            "approval_correctness": 1.0 if approval_ok else 0.0,
            "trajectory_quality": 1.0 if trajectory_ok else 0.0,
        }
        weights = {**DEFAULT_CLAW_WEIGHTS, **manifest.scoring_weights}
        total = round(sum(metrics[name] * float(weights.get(name, 0.0)) for name in DEFAULT_CLAW_WEIGHTS), 4)
        return ClawScore(total_score=total, passed=total >= 0.8 and not failure_reasons, metrics=metrics, failure_reasons=failure_reasons)
