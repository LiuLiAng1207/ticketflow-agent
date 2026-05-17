from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from .models import TicketRecord


POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    customer_tier TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    product TEXT NOT NULL,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    linked_order_id TEXT,
    expected_category TEXT,
    source_dataset TEXT,
    source_ticket_ref TEXT,
    source_language TEXT,
    source_queue TEXT,
    source_subject TEXT,
    source_body TEXT
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id BIGSERIAL PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    payload JSONB NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS workflow_tasks (
    task_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    thread_id TEXT,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    celery_task_id TEXT,
    result_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    cancelled_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS outbox_events (
    event_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    business_key TEXT NOT NULL,
    status TEXT NOT NULL,
    payload JSONB NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    delivered_at TIMESTAMPTZ,
    UNIQUE(ticket_id, operation_type, business_key)
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key_scope TEXT NOT NULL,
    key_value TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    response_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(key_scope, key_value)
);

CREATE TABLE IF NOT EXISTS external_operation_locks (
    lock_key TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    acquired_by TEXT NOT NULL,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS approval_requests (
    approval_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_args JSONB NOT NULL,
    payload JSONB NOT NULL,
    status TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    expires_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS approval_decisions (
    decision_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    comment TEXT,
    edited_action JSONB,
    created_at TIMESTAMPTZ NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _dump(payload: dict[str, object] | None) -> str:
    return json.dumps(payload or {}, ensure_ascii=False)


class PostgresTicketFlowRepository:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def init_database(self) -> None:
        with self.connect() as conn:
            conn.execute(POSTGRES_SCHEMA)
            conn.commit()

    def list_open_tickets(self, limit: int = 50) -> list[TicketRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status = 'open' ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [TicketRecord.model_validate(dict(row)) for row in rows]

    def get_ticket(self, ticket_id: str) -> TicketRecord:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE ticket_id = %s", (ticket_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown ticket_id: {ticket_id}")
        return TicketRecord.model_validate(dict(row))

    def save_audit_log(
        self,
        ticket_id: str,
        actor: str,
        event_type: str,
        detail: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO audit_log (ticket_id, actor, event_type, detail, payload)
                VALUES (%s, %s, %s, %s, %s::jsonb)
                RETURNING audit_id
                """,
                (ticket_id, actor, event_type, detail, _dump(payload)),
            ).fetchone()
            conn.commit()
        assert row is not None
        return {"audit_id": row["audit_id"]}

    def list_audit_log(self, ticket_id: str, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT audit_id, ticket_id, actor, event_type, detail, payload, created_at
                FROM audit_log
                WHERE ticket_id = %s
                ORDER BY audit_id ASC
                LIMIT %s
                """,
                (ticket_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def create_workflow_task(self, ticket_id: str, mode: str, thread_id: str | None = None) -> dict[str, object]:
        task_id = f"task-{uuid4().hex}"
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO workflow_tasks (task_id, ticket_id, thread_id, status, mode, created_at, updated_at)
                VALUES (%s, %s, %s, 'queued', %s, %s, %s)
                RETURNING *
                """,
                (task_id, ticket_id, thread_id, mode, now, now),
            ).fetchone()
            conn.commit()
        assert row is not None
        return dict(row)

    def get_workflow_task(self, task_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM workflow_tasks WHERE task_id = %s", (task_id,)).fetchone()
        return dict(row) if row is not None else None

    def create_outbox_event(
        self,
        *,
        ticket_id: str,
        operation_type: str,
        business_key: str,
        payload: dict[str, object],
    ) -> dict[str, object]:
        event_id = f"outbox-{uuid4().hex}"
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO outbox_events
                (event_id, ticket_id, operation_type, business_key, status, payload, created_at, updated_at)
                VALUES (%s, %s, %s, %s, 'pending', %s::jsonb, %s, %s)
                ON CONFLICT (ticket_id, operation_type, business_key) DO UPDATE
                SET updated_at = outbox_events.updated_at
                RETURNING *
                """,
                (event_id, ticket_id, operation_type, business_key, _dump(payload), now, now),
            ).fetchone()
            conn.commit()
        assert row is not None
        result = dict(row)
        result["deduplicated"] = result["event_id"] != event_id
        return result

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
    ) -> dict[str, object]:
        approval_id = f"approval-{uuid4().hex}"
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO approval_requests
                (approval_id, ticket_id, thread_id, tool_name, tool_args, payload, status, requested_by, created_at, updated_at, expires_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, 'pending', %s, %s, %s, %s)
                RETURNING *
                """,
                (approval_id, ticket_id, thread_id, tool_name, _dump(tool_args), _dump(payload), requested_by, now, now, expires_at),
            ).fetchone()
            conn.commit()
        assert row is not None
        return dict(row)
