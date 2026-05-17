from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from .db import TicketFlowRepository
from .models import TicketRecord
from .service_settings import ServiceSettings


@runtime_checkable
class RepositoryProtocol(Protocol):
    def init_database(self) -> None: ...

    def list_open_tickets(self, limit: int = 50) -> list[TicketRecord]: ...

    def get_ticket(self, ticket_id: str) -> TicketRecord: ...

    def create_ticket(
        self,
        *,
        title: str,
        body: str,
        customer_id: str = "CUST-001",
        customer_tier: str = "standard",
        product: str = "未指定产品",
        channel: str = "web",
        linked_order_id: str | None = None,
        expected_category: str | None = None,
    ) -> TicketRecord: ...

    def save_audit_log(
        self,
        ticket_id: str,
        actor: str,
        event_type: str,
        detail: str,
        payload: dict[str, object],
    ) -> dict[str, object]: ...

    def list_audit_log(self, ticket_id: str, limit: int = 100) -> list[dict[str, object]]: ...

    def create_workflow_task(self, ticket_id: str, mode: str, thread_id: str | None = None) -> dict[str, object]: ...

    def get_workflow_task(self, task_id: str) -> dict[str, object] | None: ...

    def set_workflow_task_celery_id(self, task_id: str, celery_task_id: str) -> dict[str, object]: ...

    def list_approval_requests(self, status: str | None = None, limit: int = 100) -> list[dict[str, object]]: ...

    def list_outbox_events(self, status: str | None = None, limit: int = 100) -> list[dict[str, object]]: ...

    def create_outbox_event(
        self,
        *,
        ticket_id: str,
        operation_type: str,
        business_key: str,
        payload: dict[str, object],
    ) -> dict[str, object]: ...

    def record_idempotency_key(
        self,
        *,
        key_scope: str,
        key_value: str,
        request_hash: str,
        response_payload: dict[str, object],
        status: str = "completed",
    ) -> dict[str, object]: ...

    def create_approval_request(
        self,
        *,
        ticket_id: str,
        thread_id: str,
        tool_name: str,
        tool_args: dict[str, object],
        payload: dict[str, object],
        requested_by: str,
        expires_at: str | None = None,
    ) -> dict[str, object]: ...

    def upsert_agent_skill(self, manifest: dict[str, object], skill_doc: str = "") -> dict[str, object]: ...

    def list_agent_skills(self, limit: int = 100) -> list[dict[str, object]]: ...

    def get_agent_skill(self, skill_id: str) -> dict[str, object] | None: ...

    def set_agent_skill_enabled(self, skill_id: str, enabled: bool) -> dict[str, object]: ...

    def create_skill_run(
        self,
        *,
        skill_id: str,
        version: str | None,
        actor: str,
        ticket_id: str | None,
        input_payload: dict[str, object],
        status: str = "running",
        result_payload: dict[str, object] | None = None,
        error_message: str | None = None,
        requires_approval: bool = False,
        idempotency_key: str | None = None,
    ) -> dict[str, object]: ...

    def update_skill_run(
        self,
        run_id: str,
        *,
        status: str,
        result_payload: dict[str, object] | None = None,
        error_message: str | None = None,
        requires_approval: bool = False,
    ) -> dict[str, object]: ...

    def list_skill_runs(self, skill_id: str | None = None, limit: int = 100) -> list[dict[str, object]]: ...

    def upsert_claw_task(self, manifest: dict[str, object], task_doc: str = "") -> dict[str, object]: ...

    def list_claw_tasks(self, limit: int = 100) -> list[dict[str, object]]: ...

    def get_claw_task(self, task_id: str) -> dict[str, object] | None: ...

    def create_claw_run(
        self,
        *,
        task_id: str,
        actor: str,
        pass_k: int,
        config_payload: dict[str, object] | None = None,
        status: str = "running",
    ) -> dict[str, object]: ...

    def update_claw_run(
        self,
        run_id: str,
        *,
        status: str,
        summary_payload: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> dict[str, object]: ...

    def create_claw_attempt(
        self,
        *,
        run_id: str,
        attempt_index: int,
        input_payload: dict[str, object] | None = None,
        status: str = "running",
    ) -> dict[str, object]: ...

    def update_claw_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        output_payload: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> dict[str, object]: ...

    def record_claw_trajectory(
        self,
        *,
        attempt_id: str,
        event_type: str,
        detail: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]: ...

    def record_claw_score(
        self,
        *,
        attempt_id: str,
        metrics: dict[str, object],
        total_score: float,
        passed: bool,
        failure_reasons: list[object] | None = None,
    ) -> dict[str, object]: ...

    def get_claw_run(self, run_id: str) -> dict[str, object] | None: ...

    def list_claw_runs(self, task_id: str | None = None, limit: int = 100) -> list[dict[str, object]]: ...

    def list_claw_leaderboard(self, limit: int = 100) -> list[dict[str, object]]: ...


def create_repository(settings: ServiceSettings, *, sqlite_db_path: Path | None = None) -> RepositoryProtocol:
    if settings.database_backend == "postgres":
        from .postgres_repository import PostgresTicketFlowRepository

        if not settings.database_url:
            raise ValueError("DATABASE_URL is required when DATABASE_BACKEND=postgres")
        return PostgresTicketFlowRepository(settings.database_url)
    return TicketFlowRepository(sqlite_db_path or settings.project_root / "data" / "generated" / "ticketflow.sqlite3")
