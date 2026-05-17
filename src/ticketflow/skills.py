from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator

from .knowledge_graph import build_ticket_graph_from_repository, create_knowledge_graph_store
from .repository import RepositoryProtocol
from .service_settings import ServiceSettings


RiskLevel = Literal["low", "medium", "elevated", "high"]


class SkillManifest(BaseModel):
    skill_id: str
    name: str
    version: str
    description: str
    risk_level: RiskLevel
    enabled: bool = True
    approval_required: bool = False
    entrypoint: str
    permissions: list[str] = Field(default_factory=list)
    tools: list[str] = Field(default_factory=list)
    owner: str = "ticketflow"

    @field_validator("skill_id", "name", "version", "description", "entrypoint")
    @classmethod
    def _not_empty(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("field must not be empty")
        return value.strip()


class SkillRunResult(BaseModel):
    run_id: str
    skill_id: str
    status: Literal["running", "succeeded", "failed", "rejected"]
    result: dict[str, Any] = Field(default_factory=dict)
    requires_approval: bool = False
    audit_events: list[dict[str, Any]] = Field(default_factory=list)
    error_message: str | None = None


def load_skill_manifest(skill_dir: Path) -> SkillManifest:
    manifest_path = skill_dir / "skill.yaml"
    if not manifest_path.exists():
        raise ValueError(f"Missing skill.yaml: {skill_dir}")
    try:
        raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8")) or {}
        return SkillManifest.model_validate(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid skill manifest {manifest_path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML in {manifest_path}: {exc}") from exc


def _skill_doc(skill_dir: Path) -> str:
    path = skill_dir / "SKILL.md"
    return path.read_text(encoding="utf-8") if path.exists() else ""


def _json_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class SkillRegistry:
    def __init__(self, repository: RepositoryProtocol, skills_dir: str | Path):
        self.repository = repository
        self.skills_dir = Path(skills_dir)

    def discover(self) -> list[SkillManifest]:
        if not self.skills_dir.exists():
            return []
        manifests: list[SkillManifest] = []
        for skill_dir in sorted(path for path in self.skills_dir.iterdir() if path.is_dir()):
            if (skill_dir / "skill.yaml").exists():
                manifests.append(load_skill_manifest(skill_dir))
        return manifests

    def reload(self) -> dict[str, Any]:
        loaded: list[dict[str, object]] = []
        for manifest in self.discover():
            skill_dir = self.skills_dir / manifest.skill_id
            stored = self.repository.upsert_agent_skill(
                manifest.model_dump(mode="json"),
                skill_doc=_skill_doc(skill_dir),
            )
            loaded.append(stored)
        return {"status": "ok", "loaded_count": len(loaded), "skills": loaded}


class SkillRuntime:
    def __init__(
        self,
        repository: RepositoryProtocol,
        *,
        project_root: str | Path | None = None,
        runner_overrides: dict[str, str] | None = None,
        knowledge_graph_store: Any | None = None,
    ):
        self.repository = repository
        self.project_root = Path(project_root) if project_root is not None else Path.cwd()
        self.runner_overrides = runner_overrides
        self.knowledge_graph_store = knowledge_graph_store

    def run_skill(
        self,
        skill_id: str,
        *,
        input_payload: dict[str, Any],
        actor: str,
        ticket_id: str | None = None,
        idempotency_key: str | None = None,
    ) -> SkillRunResult:
        skill = self.repository.get_agent_skill(skill_id)
        if skill is None:
            raise KeyError(f"Unknown skill_id: {skill_id}")
        version = str(skill.get("current_version") or "")
        if not bool(skill.get("enabled", True)):
            run = self.repository.create_skill_run(
                skill_id=skill_id,
                version=version,
                actor=actor,
                ticket_id=ticket_id,
                input_payload=input_payload,
                status="rejected",
                result_payload={},
                error_message="skill_disabled",
                idempotency_key=idempotency_key,
            )
            return self._result_from_run(run)

        if idempotency_key:
            existing = self.repository.record_idempotency_key(
                key_scope=f"skill:{skill_id}",
                key_value=idempotency_key,
                request_hash=_json_hash(input_payload),
                response_payload={},
                status="running",
            )
            if existing.get("deduplicated") and existing.get("response_payload"):
                payload = existing["response_payload"]
                if isinstance(payload, dict) and payload.get("run_id"):
                    return SkillRunResult.model_validate(payload)

        run = self.repository.create_skill_run(
            skill_id=skill_id,
            version=version,
            actor=actor,
            ticket_id=ticket_id,
            input_payload=input_payload,
            status="running",
            idempotency_key=idempotency_key,
        )
        try:
            result, requires_approval, audit_events = self._execute(skill, input_payload, ticket_id=ticket_id)
            status = "succeeded"
            error_message = None
        except Exception as exc:  # noqa: BLE001 - runtime must convert tool failures to auditable runs.
            result = {}
            requires_approval = False
            audit_events = [{"event_type": "skill_failed", "detail": str(exc)}]
            status = "failed"
            error_message = str(exc)
        updated = self.repository.update_skill_run(
            str(run["run_id"]),
            status=status,
            result_payload=result,
            error_message=error_message,
            requires_approval=requires_approval,
        )
        final = self._result_from_run(updated, audit_events=audit_events)
        if idempotency_key:
            self.repository.record_idempotency_key(
                key_scope=f"skill:{skill_id}",
                key_value=idempotency_key,
                request_hash=_json_hash(input_payload),
                response_payload=final.model_dump(mode="json"),
                status=status,
            )
        return final

    def _execute(
        self,
        skill: dict[str, object],
        input_payload: dict[str, Any],
        *,
        ticket_id: str | None,
    ) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
        skill_id = str(skill["skill_id"])
        if skill_id == "ticketflow-ops":
            return self._run_ticketflow_ops(input_payload, ticket_id=ticket_id)
        if skill_id == "ticketflow-kg-memory":
            return self._run_kg_memory(input_payload, ticket_id=ticket_id)
        if skill_id == "ticketflow-safety-governance":
            return self._run_safety_governance(input_payload, ticket_id=ticket_id)
        if skill_id == "ticketflow-batch-ops":
            return self._run_batch_ops(input_payload)
        if skill_id == "ticketflow-claw-eval":
            return self._run_claw_eval(input_payload)
        raise ValueError(f"No allowlisted executor for skill_id: {skill_id}")

    def _run_ticketflow_ops(
        self,
        input_payload: dict[str, Any],
        *,
        ticket_id: str | None,
    ) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
        operation = str(input_payload.get("operation") or ("query_ticket" if ticket_id else "list_tickets"))
        if operation == "query_ticket":
            selected_ticket_id = str(input_payload.get("ticket_id") or ticket_id)
            ticket = self.repository.get_ticket(selected_ticket_id)
            return {"ticket": ticket.model_dump(mode="json")}, False, [{"event_type": "skill_ticket_query"}]
        if operation == "list_tickets":
            limit = int(input_payload.get("limit") or 20)
            tickets = [ticket.model_dump(mode="json") for ticket in self.repository.list_open_tickets(limit=limit)]
            return {"tickets": tickets, "count": len(tickets)}, False, [{"event_type": "skill_ticket_list"}]
        if operation == "list_approvals":
            approvals = self.repository.list_approval_requests(status=input_payload.get("status"), limit=int(input_payload.get("limit") or 20))
            return {"approvals": approvals, "count": len(approvals)}, False, [{"event_type": "skill_approval_list"}]
        if operation == "list_outbox":
            events = self.repository.list_outbox_events(status=input_payload.get("status"), limit=int(input_payload.get("limit") or 20))
            return {"events": events, "count": len(events)}, False, [{"event_type": "skill_outbox_list"}]
        if operation == "run_ticket":
            selected_ticket_id = str(input_payload.get("ticket_id") or ticket_id)
            ticket = self.repository.get_ticket(selected_ticket_id)
            from .worker import enqueue_run_ticket_workflow

            task = self.repository.create_workflow_task(ticket_id=ticket.ticket_id, mode="async")
            celery_task_id = enqueue_run_ticket_workflow(
                task_id=str(task["task_id"]),
                project_root=self.project_root,
                runner_overrides=self.runner_overrides,
            )
            current_task = self.repository.get_workflow_task(str(task["task_id"])) or task
            if current_task["status"] in {"queued", "running"}:
                current_task = self.repository.set_workflow_task_celery_id(str(task["task_id"]), celery_task_id)
            return {"ticket": ticket.model_dump(mode="json"), "task": current_task}, False, [{"event_type": "skill_workflow_started"}]
        raise ValueError(f"Unsupported ticketflow-ops operation: {operation}")

    def _run_kg_memory(
        self,
        input_payload: dict[str, Any],
        *,
        ticket_id: str | None,
    ) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
        operation = str(input_payload.get("operation") or "explain_ticket")
        selected_ticket_id = str(input_payload.get("ticket_id") or ticket_id or "")
        if operation in {"explain_ticket", "rebuild_ticket_graph"}:
            if not selected_ticket_id:
                raise ValueError("ticket_id is required for KG memory operations")
            store = self.knowledge_graph_store
            should_close = False
            if store is None:
                settings = ServiceSettings.from_project_root(self.project_root, overrides=self.runner_overrides)
                store = create_knowledge_graph_store(settings)
                should_close = True
            try:
                graph = store.get_ticket_graph(selected_ticket_id)
                if operation == "rebuild_ticket_graph" or (graph.get("enabled") and not graph.get("nodes")):
                    built = build_ticket_graph_from_repository(self.repository, selected_ticket_id)
                    store.upsert_ticket_graph(built)
                    graph = store.get_ticket_graph(selected_ticket_id)
                return {"ticket_id": selected_ticket_id, "graph": graph}, False, [{"event_type": "skill_kg_explain"}]
            finally:
                if should_close:
                    store.close()
        if operation == "submit_knowledge_candidate":
            event = self.repository.create_outbox_event(
                ticket_id=selected_ticket_id or "TCK-KB-SKILL",
                operation_type="kb_candidate_email",
                business_key=f"kb_candidate_skill:{selected_ticket_id or 'general'}:{_json_hash(input_payload)[:12]}",
                payload={"source": "skill_runtime", "content": input_payload.get("content") or input_payload},
            )
            return {"event": event}, False, [{"event_type": "skill_kb_candidate_submitted"}]
        raise ValueError(f"Unsupported ticketflow-kg-memory operation: {operation}")

    def _run_safety_governance(
        self,
        input_payload: dict[str, Any],
        *,
        ticket_id: str | None,
    ) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
        selected_ticket_id = str(input_payload.get("ticket_id") or ticket_id or "")
        if not selected_ticket_id:
            raise ValueError("ticket_id is required for safety governance inspection")
        audit = self.repository.list_audit_log(selected_ticket_id, limit=int(input_payload.get("limit") or 100))
        approvals = [
            approval
            for approval in self.repository.list_approval_requests(limit=100)
            if str(approval.get("ticket_id")) == selected_ticket_id
        ]
        return {
            "ticket_id": selected_ticket_id,
            "audit_events": audit,
            "approvals": approvals,
            "message": "Governance inspection is read-only and does not approve tools.",
        }, False, [{"event_type": "skill_governance_inspected"}]

    def _run_batch_ops(self, input_payload: dict[str, Any]) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
        operation = str(input_payload.get("operation") or "batch_query")
        ticket_ids = [str(item) for item in input_payload.get("ticket_ids", [])]
        if operation == "batch_query":
            tickets = [self.repository.get_ticket(ticket_id).model_dump(mode="json") for ticket_id in ticket_ids]
            return {"tickets": tickets, "count": len(tickets)}, False, [{"event_type": "skill_batch_query"}]
        if operation == "batch_run_low_risk":
            tasks: list[dict[str, object]] = []
            for ticket_id in ticket_ids:
                ticket = self.repository.get_ticket(ticket_id)
                if ticket.expected_category in {"billing_refund", "technical_issue"} or ticket.customer_tier == "enterprise":
                    continue
                tasks.append(self.repository.create_workflow_task(ticket_id=ticket.ticket_id, mode="async"))
            return {"tasks": tasks, "count": len(tasks), "skipped_count": len(ticket_ids) - len(tasks)}, False, [{"event_type": "skill_batch_low_risk_started"}]
        raise ValueError(f"Unsupported ticketflow-batch-ops operation: {operation}")

    def _run_claw_eval(self, input_payload: dict[str, Any]) -> tuple[dict[str, Any], bool, list[dict[str, Any]]]:
        from .claw import ClawRegistry, ClawRuntime

        settings = ServiceSettings.from_project_root(self.project_root, overrides=self.runner_overrides)
        task_id = str(input_payload.get("task_id") or "claw-query-ticket-evidence")
        pass_k = int(input_payload.get("pass_k") or 1)
        ClawRegistry(self.repository, settings.claw_tasks_dir).reload()
        result = ClawRuntime(
            self.repository,
            project_root=self.project_root,
            runner_overrides=self.runner_overrides,
        ).run_task(task_id, actor=str(input_payload.get("actor") or "skill-runtime"), pass_k=pass_k)
        return {"claw_run": result.model_dump(mode="json")}, False, [{"event_type": "skill_claw_run", "task_id": task_id}]

    @staticmethod
    def _result_from_run(row: dict[str, object], audit_events: list[dict[str, Any]] | None = None) -> SkillRunResult:
        return SkillRunResult(
            run_id=str(row["run_id"]),
            skill_id=str(row["skill_id"]),
            status=str(row["status"]),  # type: ignore[arg-type]
            result=row.get("result_payload") if isinstance(row.get("result_payload"), dict) else {},
            requires_approval=bool(row.get("requires_approval")),
            audit_events=audit_events or [],
            error_message=row.get("error_message") if row.get("error_message") is not None else None,
        )
