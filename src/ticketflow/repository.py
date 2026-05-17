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

    def create_outbox_event(
        self,
        *,
        ticket_id: str,
        operation_type: str,
        business_key: str,
        payload: dict[str, object],
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


def create_repository(settings: ServiceSettings, *, sqlite_db_path: Path | None = None) -> RepositoryProtocol:
    if settings.database_backend == "postgres":
        from .postgres_repository import PostgresTicketFlowRepository

        if not settings.database_url:
            raise ValueError("DATABASE_URL is required when DATABASE_BACKEND=postgres")
        return PostgresTicketFlowRepository(settings.database_url)
    return TicketFlowRepository(sqlite_db_path or settings.project_root / "data" / "generated" / "ticketflow.sqlite3")
