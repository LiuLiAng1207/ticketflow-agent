from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.rows import dict_row

from .db import _query_terms
from .models import AttachmentEvidence, AttachmentRecord, CustomerProfile, ExternalOpRecord, OrderRecord, TicketRecord


POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS customers (
    customer_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT NOT NULL,
    customer_tier TEXT NOT NULL,
    region TEXT NOT NULL,
    loyalty_years INTEGER NOT NULL,
    open_tickets INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS orders (
    order_id TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL,
    product TEXT NOT NULL,
    status TEXT NOT NULL,
    delivered_days_ago INTEGER,
    amount DOUBLE PRECISION NOT NULL,
    eligible_for_refund BOOLEAN NOT NULL
);

CREATE TABLE IF NOT EXISTS tickets (
    ticket_id TEXT PRIMARY KEY,
    channel TEXT NOT NULL,
    customer_id TEXT NOT NULL,
    customer_tier TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    product TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
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

CREATE TABLE IF NOT EXISTS kb_articles (
    doc_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    tags TEXT NOT NULL,
    product TEXT NOT NULL,
    category TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS policies (
    policy_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    action_type TEXT NOT NULL,
    approval_required BOOLEAN NOT NULL,
    priority_hint TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reply_templates (
    template_id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    tone TEXT NOT NULL,
    body TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ticket_history (
    event_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    message TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    agent_name TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ticket_attachments (
    attachment_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    filename TEXT NOT NULL,
    file_type TEXT NOT NULL,
    source_dataset TEXT,
    storage_path TEXT,
    content_hash TEXT,
    ocr_text TEXT NOT NULL DEFAULT '',
    visual_summary TEXT NOT NULL DEFAULT '',
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    parse_status TEXT NOT NULL DEFAULT 'pending',
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS attachment_evidence (
    evidence_id TEXT PRIMARY KEY,
    attachment_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    extracted_text TEXT NOT NULL DEFAULT '',
    visual_summary TEXT NOT NULL DEFAULT '',
    entities JSONB NOT NULL DEFAULT '{}'::jsonb,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    source_span TEXT,
    bbox JSONB,
    risk_flags JSONB NOT NULL DEFAULT '[]'::jsonb,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id BIGSERIAL PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    priority TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS refund_requests (
    refund_id BIGSERIAL PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    amount DOUBLE PRECISION,
    rationale TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT now()
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

CREATE TABLE IF NOT EXISTS external_email_deliveries (
    delivery_id BIGSERIAL PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    message_type TEXT NOT NULL,
    recipient TEXT NOT NULL,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,
    provider_message_id TEXT,
    latency_ms INTEGER,
    error_message TEXT,
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

CREATE TABLE IF NOT EXISTS agent_skills (
    skill_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    approval_required BOOLEAN NOT NULL,
    current_version TEXT NOT NULL,
    manifest JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_versions (
    skill_id TEXT NOT NULL,
    version TEXT NOT NULL,
    manifest JSONB NOT NULL,
    skill_doc TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(skill_id, version)
);

CREATE TABLE IF NOT EXISTS skill_permissions (
    skill_id TEXT NOT NULL,
    permission TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    PRIMARY KEY(skill_id, permission)
);

CREATE TABLE IF NOT EXISTS skill_runs (
    run_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    version TEXT,
    actor TEXT NOT NULL,
    ticket_id TEXT,
    status TEXT NOT NULL,
    input_payload JSONB NOT NULL,
    result_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    requires_approval BOOLEAN NOT NULL DEFAULT false,
    idempotency_key TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS claw_tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    goal TEXT NOT NULL,
    category TEXT NOT NULL,
    enabled BOOLEAN NOT NULL,
    pass_k INTEGER NOT NULL,
    manifest JSONB NOT NULL,
    task_doc TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS claw_runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    status TEXT NOT NULL,
    pass_k INTEGER NOT NULL,
    config_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS claw_attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    input_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    output_payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ,
    UNIQUE(run_id, attempt_index)
);

CREATE TABLE IF NOT EXISTS claw_trajectories (
    trajectory_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    event_index INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    payload JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS claw_scores (
    score_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    metrics JSONB NOT NULL,
    total_score DOUBLE PRECISION NOT NULL,
    passed BOOLEAN NOT NULL,
    failure_reasons JSONB NOT NULL DEFAULT '[]'::jsonb,
    created_at TIMESTAMPTZ NOT NULL
);

CREATE TABLE IF NOT EXISTS observability_events (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    span_name TEXT NOT NULL,
    component TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    error_type TEXT,
    payload_summary JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(payload: dict[str, object] | list[object] | None) -> str:
    return json.dumps({} if payload is None else payload, ensure_ascii=False)


def _json_load(raw: object) -> object:
    if raw is None or raw == "":
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(str(raw))
    except json.JSONDecodeError:
        return {"raw": str(raw)}


def _wire_row(row: dict[str, object]) -> dict[str, object]:
    result = dict(row)
    for key, value in list(result.items()):
        if isinstance(value, datetime):
            result[key] = value.isoformat()
    return result


class PostgresTicketFlowRepository:
    def __init__(self, database_url: str):
        self.database_url = database_url

    def connect(self):
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def init_database(self) -> None:
        with self.connect() as conn:
            for statement in POSTGRES_SCHEMA.split(";"):
                statement = statement.strip()
                if statement:
                    conn.execute(statement)
            conn.commit()

    @staticmethod
    def _coerce_value(value: str) -> object:
        if value == "":
            return None
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        try:
            return int(value)
        except ValueError:
            pass
        try:
            return float(value)
        except ValueError:
            return value

    def load_seed_directory(self, seed_dir: Path) -> None:
        self.init_database()
        tables = {
            "customers.csv": "customers",
            "orders.csv": "orders",
            "tickets.csv": "tickets",
            "ticket_attachments.csv": "ticket_attachments",
            "kb_articles.csv": "kb_articles",
            "policies.csv": "policies",
            "reply_templates.csv": "reply_templates",
            "ticket_history.csv": "ticket_history",
        }
        with self.connect() as conn:
            for filename, table in tables.items():
                path = seed_dir / filename
                if not path.exists():
                    continue
                with path.open("r", encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                if not rows:
                    continue
                conn.execute(f"DELETE FROM {table}")
                columns = list(rows[0].keys())
                placeholders = ", ".join(["%s"] * len(columns))
                with conn.cursor() as cursor:
                    cursor.executemany(
                        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                        [tuple(self._coerce_value(row[column]) for column in columns) for row in rows],
                    )
            conn.commit()

    def bootstrap(self, seed_dir: Path) -> None:
        self.init_database()
        with self.connect() as conn:
            count = conn.execute("SELECT COUNT(*) AS count FROM tickets").fetchone()["count"]
        if count == 0:
            self.load_seed_directory(seed_dir)

    def reset_from_seed(self, seed_dir: Path) -> None:
        self.load_seed_directory(seed_dir)
        generated_tables = [
            "attachment_evidence",
            "escalations",
            "refund_requests",
            "audit_log",
            "external_email_deliveries",
            "observability_events",
            "workflow_tasks",
            "outbox_events",
            "idempotency_keys",
            "external_operation_locks",
            "skill_runs",
            "approval_decisions",
            "approval_requests",
        ]
        with self.connect() as conn:
            for table in generated_tables:
                conn.execute(f"DELETE FROM {table}")
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
    ) -> TicketRecord:
        ticket_id = f"TCK-CHAT-{uuid4().hex[:8].upper()}"
        created_at = datetime.now(timezone.utc).isoformat()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO tickets
                    (ticket_id, channel, customer_id, customer_tier, title, body, product, created_at, status,
                     linked_order_id, expected_category, source_dataset, source_ticket_ref, source_language,
                     source_queue, source_subject, source_body)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'open', %s, %s, 'agent_chat', %s, 'zh', 'agent_chat', %s, %s)
                """,
                (
                    ticket_id,
                    channel,
                    customer_id,
                    customer_tier,
                    title,
                    body,
                    product,
                    created_at,
                    linked_order_id,
                    expected_category,
                    ticket_id,
                    title,
                    body,
                ),
            )
            conn.commit()
        return self.get_ticket(ticket_id)

    def get_customer_profile(self, customer_id: str) -> CustomerProfile | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM customers WHERE customer_id = %s", (customer_id,)).fetchone()
        return CustomerProfile.model_validate(dict(row)) if row else None

    def get_order_status(self, order_id: str | None) -> OrderRecord | None:
        if not order_id:
            return None
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM orders WHERE order_id = %s", (order_id,)).fetchone()
        return OrderRecord.model_validate(dict(row)) if row else None

    def get_ticket_history(self, ticket_id: str, limit: int = 5) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT event_id, ticket_id, message, created_at, agent_name
                FROM ticket_history
                WHERE ticket_id = %s
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (ticket_id, limit),
            ).fetchall()
        return [_wire_row(dict(row)) for row in rows]

    def create_ticket_attachment(
        self,
        *,
        ticket_id: str,
        filename: str,
        file_type: str = "unknown",
        source_dataset: str | None = None,
        storage_path: str | None = None,
        content_hash: str | None = None,
        ocr_text: str = "",
        visual_summary: str = "",
        metadata: dict[str, object] | None = None,
    ) -> AttachmentRecord:
        metadata = metadata or {}
        if content_hash is None:
            content_hash = hashlib.sha256(f"{filename}\n{ocr_text}\n{visual_summary}".encode("utf-8")).hexdigest()
        attachment_id = f"ATT-{uuid4().hex[:12]}"
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO ticket_attachments
                    (attachment_id, ticket_id, filename, file_type, source_dataset, storage_path, content_hash,
                     ocr_text, visual_summary, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                """,
                (
                    attachment_id,
                    ticket_id,
                    filename,
                    file_type,
                    source_dataset,
                    storage_path,
                    content_hash,
                    ocr_text,
                    visual_summary,
                    _json_dump(metadata),
                ),
            )
            conn.commit()
        return self.get_ticket_attachment(attachment_id)

    def get_ticket_attachment(self, attachment_id: str) -> AttachmentRecord:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM ticket_attachments WHERE attachment_id = %s", (attachment_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown attachment_id: {attachment_id}")
        return self._attachment_from_row(dict(row))

    def list_ticket_attachments(self, ticket_id: str) -> list[AttachmentRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ticket_attachments WHERE ticket_id = %s ORDER BY created_at ASC, attachment_id ASC",
                (ticket_id,),
            ).fetchall()
        return [self._attachment_from_row(dict(row)) for row in rows]

    def replace_attachment_evidence(self, ticket_id: str, evidence: list[AttachmentEvidence]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM attachment_evidence WHERE ticket_id = %s", (ticket_id,))
            for item in evidence:
                conn.execute(
                    """
                    INSERT INTO attachment_evidence
                        (evidence_id, attachment_id, ticket_id, evidence_type, extracted_text, visual_summary,
                         entities, confidence, source_span, bbox, risk_flags, metadata)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb)
                    """,
                    (
                        item.evidence_id,
                        item.attachment_id,
                        item.ticket_id,
                        item.evidence_type,
                        item.extracted_text,
                        item.visual_summary,
                        _json_dump(item.entities),
                        item.confidence,
                        item.source_span,
                        _json_dump(item.bbox) if item.bbox is not None else None,
                        _json_dump(item.risk_flags),
                        _json_dump(item.metadata),
                    ),
                )
            conn.commit()

    def update_attachment_parse_status(self, attachment_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE ticket_attachments SET parse_status = %s WHERE attachment_id = %s",
                (status, attachment_id),
            )
            conn.commit()

    def list_attachment_evidence(self, ticket_id: str) -> list[AttachmentEvidence]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM attachment_evidence WHERE ticket_id = %s ORDER BY evidence_id ASC",
                (ticket_id,),
            ).fetchall()
        return [self._attachment_evidence_from_row(dict(row)) for row in rows]

    @staticmethod
    def _attachment_from_row(row: dict[str, object]) -> AttachmentRecord:
        payload = _wire_row(row)
        payload["metadata"] = _json_load(payload.get("metadata"))
        return AttachmentRecord.model_validate(payload)

    @staticmethod
    def _attachment_evidence_from_row(row: dict[str, object]) -> AttachmentEvidence:
        payload = _wire_row(row)
        payload["entities"] = _json_load(payload.get("entities"))
        payload["bbox"] = _json_load(payload.get("bbox")) if payload.get("bbox") else None
        payload["risk_flags"] = _json_load(payload.get("risk_flags")) or []
        payload["metadata"] = _json_load(payload.get("metadata"))
        return AttachmentEvidence.model_validate(payload)

    def list_kb_articles(self) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM kb_articles").fetchall()
        return [dict(row) for row in rows]

    def list_policies(self) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM policies").fetchall()
        return [dict(row) for row in rows]

    def list_ticket_history_entries(self) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT event_id, ticket_id, message, created_at, agent_name FROM ticket_history"
            ).fetchall()
        return [_wire_row(dict(row)) for row in rows]

    def search_kb(self, query: str, product: str, limit: int = 5) -> list[dict[str, object]]:
        query_terms = _query_terms(query)
        product_prefix = product.split()[0].lower() if product.split() else product.lower()
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kb_articles WHERE product = %s OR LOWER(category) LIKE %s",
                (product, f"%{product_prefix}%"),
            ).fetchall()
        scored_rows: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            item = dict(row)
            text = f"{item['title']} {item['body']} {item['tags']}".lower()
            score = sum(1 for term in query_terms if term in text)
            if item["product"] == product:
                score += 1.5
            if score:
                scored_rows.append((score, item))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [row | {"score": score} for score, row in scored_rows[:limit]]

    def search_policy_text(self, query: str, limit: int = 5) -> list[dict[str, object]]:
        terms = _query_terms(query)
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM policies").fetchall()
        scored_rows: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            item = dict(row)
            text = f"{item['title']} {item['body']} {item['action_type']}".lower()
            score = sum(1 for term in terms if term in text)
            if item["approval_required"]:
                score += 0.2
            if score:
                scored_rows.append((score, item))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [row | {"score": score} for score, row in scored_rows[:limit]]

    def search_related_history(self, query: str, limit: int = 5) -> list[dict[str, object]]:
        terms = _query_terms(query)
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    h.event_id,
                    h.ticket_id,
                    h.message,
                    h.created_at,
                    h.agent_name,
                    t.title,
                    t.body,
                    t.product,
                    t.expected_category
                FROM ticket_history h
                JOIN tickets t ON t.ticket_id = h.ticket_id
                """
            ).fetchall()
        scored_rows: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            item = _wire_row(dict(row))
            text = " ".join(
                str(part or "")
                for part in (
                    item["message"],
                    item["agent_name"],
                    item["title"],
                    item["body"],
                    item["product"],
                    item["expected_category"],
                )
            ).lower()
            score = sum(1 for term in terms if term in text)
            if item["expected_category"] and any(term in str(item["expected_category"]).lower() for term in terms):
                score += 1.0
            if item["product"] and any(term in str(item["product"]).lower() for term in terms):
                score += 0.8
            if item["title"] and any(term in str(item["title"]).lower() for term in terms):
                score += 0.5
            if score:
                scored_rows.append((score, item))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [row | {"score": score} for score, row in scored_rows[:limit]]

    def lookup_policy(self, action_type: str, query: str = "", limit: int = 5) -> list[dict[str, object]]:
        terms = _query_terms(query)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM policies WHERE action_type = %s OR body ILIKE %s",
                (action_type, f"%{action_type}%"),
            ).fetchall()
        scored_rows: list[tuple[float, dict[str, object]]] = []
        for row in rows:
            item = dict(row)
            text = f"{item['title']} {item['body']} {item['action_type']}".lower()
            score = 1.0 if item["action_type"] == action_type else 0.0
            score += sum(1 for term in terms if term in text)
            scored_rows.append((score, item))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [row | {"score": score} for score, row in scored_rows[:limit]]

    def get_reply_templates(self, category: str, limit: int = 3) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT body FROM reply_templates WHERE category = %s LIMIT %s",
                (category, limit),
            ).fetchall()
        return [str(row["body"]) for row in rows]

    def create_escalation(self, ticket_id: str, reason: str, priority: str) -> dict[str, object]:
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO escalations (ticket_id, reason, priority)
                VALUES (%s, %s, %s)
                RETURNING escalation_id
                """,
                (ticket_id, reason, priority),
            ).fetchone()
            conn.commit()
        assert row is not None
        return {"escalation_id": row["escalation_id"], "ticket_id": ticket_id, "priority": priority}

    def issue_refund_request(self, ticket_id: str, order_id: str, amount: float, rationale: str) -> dict[str, object]:
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO refund_requests (ticket_id, order_id, amount, rationale)
                VALUES (%s, %s, %s, %s)
                RETURNING refund_id
                """,
                (ticket_id, order_id, amount, rationale),
            ).fetchone()
            conn.commit()
        assert row is not None
        return {"refund_id": row["refund_id"], "ticket_id": ticket_id, "order_id": order_id, "amount": amount}

    def update_ticket_status(self, ticket_id: str, status: str) -> dict[str, object]:
        with self.connect() as conn:
            conn.execute("UPDATE tickets SET status = %s WHERE ticket_id = %s", (status, ticket_id))
            conn.commit()
        return {"ticket_id": ticket_id, "status": status}

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
                (ticket_id, actor, event_type, detail, _json_dump(payload)),
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
        return [
            {
                **_wire_row(dict(row)),
                "payload": _json_load(row["payload"]),
            }
            for row in rows
        ]

    @staticmethod
    def _observability_event_from_row(row: dict[str, object]) -> dict[str, object]:
        return {
            **_wire_row(row),
            "payload_summary": _json_load(row.get("payload_summary")),
        }

    def record_observability_event(
        self,
        *,
        trace_id: str,
        span_name: str,
        component: str,
        status: str,
        latency_ms: int = 0,
        error_type: str | None = None,
        payload_summary: dict[str, object] | None = None,
    ) -> dict[str, object]:
        event_id = f"obs-{uuid4().hex}"
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO observability_events
                (event_id, trace_id, span_name, component, status, latency_ms, error_type, payload_summary, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING *
                """,
                (
                    event_id,
                    trace_id,
                    span_name,
                    component,
                    status,
                    int(latency_ms),
                    error_type,
                    _json_dump(payload_summary or {}),
                    now,
                ),
            ).fetchone()
            conn.commit()
        assert row is not None
        return self._observability_event_from_row(dict(row))

    def list_observability_events(self, trace_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if trace_id is None:
                rows = conn.execute(
                    "SELECT * FROM observability_events ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM observability_events WHERE trace_id = %s ORDER BY created_at ASC LIMIT %s",
                    (trace_id, limit),
                ).fetchall()
        return [self._observability_event_from_row(dict(row)) for row in rows]

    @staticmethod
    def _workflow_task_from_row(row: dict[str, object]) -> dict[str, object]:
        payload = _wire_row(row)
        payload["result"] = _json_load(payload.pop("result_json", None))
        return payload

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
        return self._workflow_task_from_row(dict(row))

    def get_workflow_task(self, task_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM workflow_tasks WHERE task_id = %s", (task_id,)).fetchone()
        return self._workflow_task_from_row(dict(row)) if row else None

    def get_workflow_task_by_thread(self, thread_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_tasks WHERE thread_id = %s ORDER BY created_at DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        return self._workflow_task_from_row(dict(row)) if row else None

    def list_workflow_tasks(self, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM workflow_tasks ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
        return [self._workflow_task_from_row(dict(row)) for row in rows]

    def mark_workflow_task_running(self, task_id: str, celery_task_id: str | None = None) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'running', celery_task_id = COALESCE(%s, celery_task_id),
                    started_at = COALESCE(started_at, %s), updated_at = %s
                WHERE task_id = %s
                RETURNING *
                """,
                (celery_task_id, now, now, task_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return self._workflow_task_from_row(dict(row))

    def set_workflow_task_celery_id(self, task_id: str, celery_task_id: str) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "UPDATE workflow_tasks SET celery_task_id = %s, updated_at = %s WHERE task_id = %s RETURNING *",
                (celery_task_id, now, task_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return self._workflow_task_from_row(dict(row))

    def complete_workflow_task(self, task_id: str, result: dict[str, object]) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'succeeded', result_json = %s::jsonb, error_message = NULL,
                    completed_at = %s, updated_at = %s
                WHERE task_id = %s
                RETURNING *
                """,
                (_json_dump(result), now, now, task_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return self._workflow_task_from_row(dict(row))

    def wait_workflow_task_for_approval(self, task_id: str, thread_id: str, approval_id: str | None) -> dict[str, object]:
        now = _utc_now()
        result = {"interrupted": True, "approval_id": approval_id, "thread_id": thread_id}
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'waiting_approval', thread_id = %s, result_json = %s::jsonb,
                    error_message = NULL, updated_at = %s
                WHERE task_id = %s
                RETURNING *
                """,
                (thread_id, _json_dump(result), now, task_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        task = self._workflow_task_from_row(dict(row))
        task["approval_id"] = approval_id
        return task

    def fail_workflow_task(self, task_id: str, error_message: str) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'failed', error_message = %s, completed_at = %s, updated_at = %s
                WHERE task_id = %s
                RETURNING *
                """,
                (error_message, now, now, task_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return self._workflow_task_from_row(dict(row))

    def cancel_workflow_task(self, task_id: str, reason: str = "cancelled") -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'cancelled', error_message = %s, cancelled_at = %s, updated_at = %s
                WHERE task_id = %s
                RETURNING *
                """,
                (reason, now, now, task_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return self._workflow_task_from_row(dict(row))

    @staticmethod
    def _outbox_event_from_row(row: dict[str, object], *, deduplicated: bool = False) -> dict[str, object]:
        payload = _wire_row(row)
        payload["payload"] = _json_load(payload.get("payload"))
        payload["deduplicated"] = deduplicated
        return payload

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
                (event_id, ticket_id, operation_type, business_key, _json_dump(payload), now, now),
            ).fetchone()
            conn.commit()
        assert row is not None
        result = self._outbox_event_from_row(dict(row))
        result["deduplicated"] = result["event_id"] != event_id
        return result

    def get_outbox_event(self, event_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM outbox_events WHERE event_id = %s", (event_id,)).fetchone()
        return self._outbox_event_from_row(dict(row)) if row else None

    def list_outbox_events(self, status: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if status is None:
                rows = conn.execute("SELECT * FROM outbox_events ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM outbox_events WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                    (status, limit),
                ).fetchall()
        return [self._outbox_event_from_row(dict(row)) for row in rows]

    def mark_outbox_event_delivered(self, event_id: str) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE outbox_events
                SET status = 'delivered', delivered_at = %s, updated_at = %s
                WHERE event_id = %s
                RETURNING *
                """,
                (now, now, event_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        return self._outbox_event_from_row(dict(row))

    def mark_outbox_event_failed(self, event_id: str, error_message: str) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE outbox_events
                SET status = 'failed', attempts = attempts + 1, last_error = %s, updated_at = %s
                WHERE event_id = %s
                RETURNING *
                """,
                (error_message, now, event_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        return self._outbox_event_from_row(dict(row))

    def record_idempotency_key(
        self,
        *,
        key_scope: str,
        key_value: str,
        request_hash: str,
        response_payload: dict[str, object],
        status: str = "completed",
    ) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO idempotency_keys
                (key_scope, key_value, request_hash, status, response_payload, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT (key_scope, key_value) DO UPDATE
                SET updated_at = idempotency_keys.updated_at
                RETURNING *
                """,
                (key_scope, key_value, request_hash, status, _json_dump(response_payload), now, now),
            ).fetchone()
            conn.commit()
        assert row is not None
        result = _wire_row(dict(row))
        result["response_payload"] = _json_load(result.get("response_payload"))
        result["deduplicated"] = result["created_at"] != now
        return result

    def acquire_external_operation_lock(
        self,
        *,
        lock_key: str,
        ticket_id: str,
        operation_type: str,
        acquired_by: str,
        expires_at: str | None = None,
    ) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO external_operation_locks
                (lock_key, ticket_id, operation_type, acquired_by, expires_at, created_at)
                VALUES (%s, %s, %s, %s, %s, %s)
                ON CONFLICT (lock_key) DO NOTHING
                RETURNING *
                """,
                (lock_key, ticket_id, operation_type, acquired_by, expires_at, now),
            ).fetchone()
            acquired = row is not None
            if row is None:
                row = conn.execute("SELECT * FROM external_operation_locks WHERE lock_key = %s", (lock_key,)).fetchone()
            conn.commit()
        assert row is not None
        result = _wire_row(dict(row))
        result["acquired"] = acquired
        return result

    @staticmethod
    def _approval_from_row(row: dict[str, object]) -> dict[str, object]:
        payload = _wire_row(row)
        payload["tool_args"] = _json_load(payload.get("tool_args"))
        payload["payload"] = _json_load(payload.get("payload"))
        return payload

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
                (approval_id, ticket_id, thread_id, tool_name, tool_args, payload, status, requested_by,
                 created_at, updated_at, expires_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, 'pending', %s, %s, %s, %s)
                RETURNING *
                """,
                (approval_id, ticket_id, thread_id, tool_name, _json_dump(tool_args), _json_dump(payload), requested_by, now, now, expires_at),
            ).fetchone()
            conn.commit()
        assert row is not None
        return self._approval_from_row(dict(row))

    def ensure_approval_request(
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
        with self.connect() as conn:
            row = conn.execute(
                """
                SELECT * FROM approval_requests
                WHERE thread_id = %s AND tool_name = %s
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (thread_id, tool_name),
            ).fetchone()
        if row is not None:
            approval = self._approval_from_row(dict(row))
            approval["deduplicated"] = True
            return approval
        approval = self.create_approval_request(
            ticket_id=ticket_id,
            thread_id=thread_id,
            tool_name=tool_name,
            tool_args=tool_args,
            payload=payload,
            requested_by=requested_by,
            expires_at=expires_at,
        )
        approval["deduplicated"] = False
        return approval

    def get_approval_request(self, approval_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM approval_requests WHERE approval_id = %s", (approval_id,)).fetchone()
        return self._approval_from_row(dict(row)) if row else None

    def list_approval_requests(self, status: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM approval_requests ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM approval_requests WHERE status = %s ORDER BY created_at DESC LIMIT %s",
                    (status, limit),
                ).fetchall()
        return [self._approval_from_row(dict(row)) for row in rows]

    def record_approval_decision(
        self,
        *,
        approval_id: str,
        decision: str,
        reviewer: str,
        comment: str | None = None,
        edited_action: dict[str, object] | None = None,
    ) -> dict[str, object]:
        status_by_decision = {"approve": "approved", "edit": "edited", "reject": "rejected"}
        request_status = status_by_decision.get(decision, decision)
        decision_id = f"decision-{uuid4().hex}"
        now = _utc_now()
        with self.connect() as conn:
            existing = conn.execute("SELECT * FROM approval_requests WHERE approval_id = %s", (approval_id,)).fetchone()
            if existing is None:
                raise KeyError(f"Unknown approval_id: {approval_id}")
            row = conn.execute(
                """
                INSERT INTO approval_decisions
                (decision_id, approval_id, decision, reviewer, comment, edited_action, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING *
                """,
                (decision_id, approval_id, decision, reviewer, comment, _json_dump(edited_action), now),
            ).fetchone()
            conn.execute(
                "UPDATE approval_requests SET status = %s, updated_at = %s WHERE approval_id = %s",
                (request_status, now, approval_id),
            )
            conn.commit()
        assert row is not None
        result = _wire_row(dict(row))
        result["edited_action"] = _json_load(result.get("edited_action"))
        return result

    def create_external_email_delivery(
        self,
        *,
        ticket_id: str,
        message_type: str,
        recipient: str,
        subject: str,
        status: str,
        provider_message_id: str | None,
        latency_ms: int | None,
        error_message: str | None,
        payload: dict[str, object],
    ) -> dict[str, object]:
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO external_email_deliveries
                    (ticket_id, message_type, recipient, subject, status, provider_message_id, latency_ms,
                     error_message, payload)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING delivery_id
                """,
                (
                    ticket_id,
                    message_type,
                    recipient,
                    subject,
                    status,
                    provider_message_id,
                    latency_ms,
                    error_message,
                    _json_dump(payload),
                ),
            ).fetchone()
            conn.commit()
        assert row is not None
        return {"delivery_id": row["delivery_id"]}

    def list_external_email_deliveries(self, ticket_id: str) -> list[ExternalOpRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT delivery_id, message_type, recipient, subject, status, provider_message_id,
                       latency_ms, error_message, payload, created_at
                FROM external_email_deliveries
                WHERE ticket_id = %s
                ORDER BY delivery_id ASC
                """,
                (ticket_id,),
            ).fetchall()
        records: list[ExternalOpRecord] = []
        for row in rows:
            payload = _json_load(row["payload"])
            records.append(
                ExternalOpRecord(
                    op_type="incident_email" if row["message_type"] == "incident_email" else "kb_candidate_email",
                    status=row["status"],
                    recipient=row["recipient"],
                    subject=row["subject"],
                    provider_message_id=row["provider_message_id"],
                    delivery_id=str(row["delivery_id"]),
                    latency_ms=row["latency_ms"],
                    error_message=row["error_message"],
                    created_at=row["created_at"],
                    payload=payload if isinstance(payload, dict) else {},
                )
            )
        return records

    @staticmethod
    def _skill_from_row(row: dict[str, object]) -> dict[str, object]:
        result = _wire_row(row)
        result["enabled"] = bool(result["enabled"])
        result["approval_required"] = bool(result["approval_required"])
        result["manifest"] = _json_load(result.get("manifest"))
        return result

    @staticmethod
    def _skill_run_from_row(row: dict[str, object]) -> dict[str, object]:
        result = _wire_row(row)
        result["requires_approval"] = bool(result["requires_approval"])
        result["input_payload"] = _json_load(result.get("input_payload"))
        result["result_payload"] = _json_load(result.get("result_payload"))
        return result

    def upsert_agent_skill(self, manifest: dict[str, object], skill_doc: str = "") -> dict[str, object]:
        now = _utc_now()
        skill_id = str(manifest["skill_id"])
        version = str(manifest["version"])
        permissions = [str(item) for item in manifest.get("permissions", [])]
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO agent_skills
                (skill_id, name, description, risk_level, enabled, approval_required, current_version, manifest, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                ON CONFLICT(skill_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    risk_level = excluded.risk_level,
                    approval_required = excluded.approval_required,
                    current_version = excluded.current_version,
                    manifest = excluded.manifest,
                    updated_at = excluded.updated_at
                RETURNING *
                """,
                (
                    skill_id,
                    str(manifest["name"]),
                    str(manifest["description"]),
                    str(manifest["risk_level"]),
                    bool(manifest.get("enabled", True)),
                    bool(manifest.get("approval_required", False)),
                    version,
                    _json_dump(manifest),
                    now,
                    now,
                ),
            ).fetchone()
            conn.execute(
                """
                INSERT INTO skill_versions (skill_id, version, manifest, skill_doc, created_at)
                VALUES (%s, %s, %s::jsonb, %s, %s)
                ON CONFLICT(skill_id, version) DO UPDATE SET
                    manifest = excluded.manifest,
                    skill_doc = excluded.skill_doc
                """,
                (skill_id, version, _json_dump(manifest), skill_doc, now),
            )
            conn.execute("DELETE FROM skill_permissions WHERE skill_id = %s", (skill_id,))
            if permissions:
                with conn.cursor() as cursor:
                    cursor.executemany(
                        "INSERT INTO skill_permissions (skill_id, permission, created_at) VALUES (%s, %s, %s)",
                        [(skill_id, permission, now) for permission in permissions],
                    )
            conn.commit()
        assert row is not None
        return self._skill_from_row(dict(row))

    def list_agent_skills(self, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_skills ORDER BY skill_id ASC LIMIT %s",
                (limit,),
            ).fetchall()
        return [self._skill_from_row(dict(row)) for row in rows]

    def get_agent_skill(self, skill_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM agent_skills WHERE skill_id = %s", (skill_id,)).fetchone()
        return self._skill_from_row(dict(row)) if row else None

    def set_agent_skill_enabled(self, skill_id: str, enabled: bool) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                "UPDATE agent_skills SET enabled = %s, updated_at = %s WHERE skill_id = %s RETURNING *",
                (enabled, now, skill_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown skill_id: {skill_id}")
        return self._skill_from_row(dict(row))

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
    ) -> dict[str, object]:
        now = _utc_now()
        run_id = f"skillrun-{uuid4().hex}"
        completed_at = now if status in {"succeeded", "failed", "rejected"} else None
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO skill_runs
                (run_id, skill_id, version, actor, ticket_id, status, input_payload, result_payload, error_message,
                 requires_approval, idempotency_key, created_at, updated_at, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s)
                RETURNING *
                """,
                (
                    run_id,
                    skill_id,
                    version,
                    actor,
                    ticket_id,
                    status,
                    _json_dump(input_payload),
                    _json_dump(result_payload),
                    error_message,
                    requires_approval,
                    idempotency_key,
                    now,
                    now,
                    completed_at,
                ),
            ).fetchone()
            conn.commit()
        assert row is not None
        return self._skill_run_from_row(dict(row))

    def update_skill_run(
        self,
        run_id: str,
        *,
        status: str,
        result_payload: dict[str, object] | None = None,
        error_message: str | None = None,
        requires_approval: bool = False,
    ) -> dict[str, object]:
        now = _utc_now()
        completed_at = now if status in {"succeeded", "failed", "rejected"} else None
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE skill_runs
                SET status = %s, result_payload = %s::jsonb, error_message = %s, requires_approval = %s,
                    updated_at = %s, completed_at = %s
                WHERE run_id = %s
                RETURNING *
                """,
                (status, _json_dump(result_payload), error_message, requires_approval, now, completed_at, run_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self._skill_run_from_row(dict(row))

    def list_skill_runs(self, skill_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if skill_id is None:
                rows = conn.execute(
                    "SELECT * FROM skill_runs ORDER BY created_at DESC LIMIT %s",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM skill_runs WHERE skill_id = %s ORDER BY created_at DESC LIMIT %s",
                    (skill_id, limit),
                ).fetchall()
        return [self._skill_run_from_row(dict(row)) for row in rows]

    @staticmethod
    def _claw_task_from_row(row: dict[str, object]) -> dict[str, object]:
        result = _wire_row(row)
        result["enabled"] = bool(result["enabled"])
        result["manifest"] = _json_load(result.get("manifest"))
        return result

    @staticmethod
    def _claw_run_from_row(row: dict[str, object]) -> dict[str, object]:
        result = _wire_row(row)
        result["config_payload"] = _json_load(result.get("config_payload"))
        result["summary"] = _json_load(result.pop("summary_payload", {}))
        return result

    @staticmethod
    def _claw_attempt_from_row(row: dict[str, object]) -> dict[str, object]:
        result = _wire_row(row)
        result["input_payload"] = _json_load(result.get("input_payload"))
        result["output_payload"] = _json_load(result.get("output_payload"))
        return result

    @staticmethod
    def _claw_trajectory_from_row(row: dict[str, object]) -> dict[str, object]:
        result = _wire_row(row)
        result["payload"] = _json_load(result.get("payload"))
        return result

    @staticmethod
    def _claw_score_from_row(row: dict[str, object]) -> dict[str, object]:
        result = _wire_row(row)
        result["metrics"] = _json_load(result.get("metrics"))
        result["passed"] = bool(result["passed"])
        result["failure_reasons"] = _json_load(result.get("failure_reasons"))
        return result

    def upsert_claw_task(self, manifest: dict[str, object], task_doc: str = "") -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO claw_tasks
                (task_id, name, goal, category, enabled, pass_k, manifest, task_doc, created_at, updated_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s)
                ON CONFLICT(task_id) DO UPDATE SET
                    name = excluded.name,
                    goal = excluded.goal,
                    category = excluded.category,
                    enabled = excluded.enabled,
                    pass_k = excluded.pass_k,
                    manifest = excluded.manifest,
                    task_doc = excluded.task_doc,
                    updated_at = excluded.updated_at
                RETURNING *
                """,
                (
                    str(manifest["task_id"]),
                    str(manifest["name"]),
                    str(manifest["goal"]),
                    str(manifest.get("category") or "ticketflow"),
                    bool(manifest.get("enabled", True)),
                    int(manifest.get("pass_k") or 3),
                    _json_dump(manifest),
                    task_doc,
                    now,
                    now,
                ),
            ).fetchone()
            conn.commit()
        assert row is not None
        return self._claw_task_from_row(dict(row))

    def list_claw_tasks(self, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM claw_tasks ORDER BY task_id ASC LIMIT %s", (limit,)).fetchall()
        return [self._claw_task_from_row(dict(row)) for row in rows]

    def get_claw_task(self, task_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM claw_tasks WHERE task_id = %s", (task_id,)).fetchone()
        return self._claw_task_from_row(dict(row)) if row else None

    def create_claw_run(
        self,
        *,
        task_id: str,
        actor: str,
        pass_k: int,
        config_payload: dict[str, object] | None = None,
        status: str = "running",
    ) -> dict[str, object]:
        now = _utc_now()
        run_id = f"clawrun-{uuid4().hex}"
        completed_at = now if status in {"succeeded", "failed"} else None
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO claw_runs
                (run_id, task_id, actor, status, pass_k, config_payload, summary_payload, error_message,
                 created_at, updated_at, completed_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
                RETURNING *
                """,
                (run_id, task_id, actor, status, pass_k, _json_dump(config_payload), _json_dump({}), None, now, now, completed_at),
            ).fetchone()
            conn.commit()
        assert row is not None
        return self._claw_run_from_row(dict(row))

    def update_claw_run(
        self,
        run_id: str,
        *,
        status: str,
        summary_payload: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> dict[str, object]:
        now = _utc_now()
        completed_at = now if status in {"succeeded", "failed"} else None
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE claw_runs
                SET status = %s, summary_payload = %s::jsonb, error_message = %s, updated_at = %s, completed_at = %s
                WHERE run_id = %s
                RETURNING *
                """,
                (status, _json_dump(summary_payload), error_message, now, completed_at, run_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown claw run_id: {run_id}")
        return self._claw_run_from_row(dict(row))

    def create_claw_attempt(
        self,
        *,
        run_id: str,
        attempt_index: int,
        input_payload: dict[str, object] | None = None,
        status: str = "running",
    ) -> dict[str, object]:
        now = _utc_now()
        attempt_id = f"clawattempt-{uuid4().hex}"
        completed_at = now if status in {"succeeded", "failed"} else None
        with self.connect() as conn:
            row = conn.execute(
                """
                INSERT INTO claw_attempts
                (attempt_id, run_id, attempt_index, status, input_payload, output_payload, error_message,
                 created_at, updated_at, completed_at)
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s)
                RETURNING *
                """,
                (attempt_id, run_id, attempt_index, status, _json_dump(input_payload), _json_dump({}), None, now, now, completed_at),
            ).fetchone()
            conn.commit()
        assert row is not None
        return self._claw_attempt_from_row(dict(row))

    def update_claw_attempt(
        self,
        attempt_id: str,
        *,
        status: str,
        output_payload: dict[str, object] | None = None,
        error_message: str | None = None,
    ) -> dict[str, object]:
        now = _utc_now()
        completed_at = now if status in {"succeeded", "failed"} else None
        with self.connect() as conn:
            row = conn.execute(
                """
                UPDATE claw_attempts
                SET status = %s, output_payload = %s::jsonb, error_message = %s, updated_at = %s, completed_at = %s
                WHERE attempt_id = %s
                RETURNING *
                """,
                (status, _json_dump(output_payload), error_message, now, completed_at, attempt_id),
            ).fetchone()
            conn.commit()
        if row is None:
            raise KeyError(f"Unknown claw attempt_id: {attempt_id}")
        return self._claw_attempt_from_row(dict(row))

    def record_claw_trajectory(
        self,
        *,
        attempt_id: str,
        event_type: str,
        detail: str,
        payload: dict[str, object] | None = None,
    ) -> dict[str, object]:
        now = _utc_now()
        trajectory_id = f"clawtraj-{uuid4().hex}"
        with self.connect() as conn:
            index_row = conn.execute(
                "SELECT COALESCE(MAX(event_index), 0) + 1 AS event_index FROM claw_trajectories WHERE attempt_id = %s",
                (attempt_id,),
            ).fetchone()
            event_index = int(index_row["event_index"] if index_row else 1)
            row = conn.execute(
                """
                INSERT INTO claw_trajectories
                (trajectory_id, attempt_id, event_index, event_type, detail, payload, created_at)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s)
                RETURNING *
                """,
                (trajectory_id, attempt_id, event_index, event_type, detail, _json_dump(payload), now),
            ).fetchone()
            conn.commit()
        assert row is not None
        return self._claw_trajectory_from_row(dict(row))

    def record_claw_score(
        self,
        *,
        attempt_id: str,
        metrics: dict[str, object],
        total_score: float,
        passed: bool,
        failure_reasons: list[object] | None = None,
    ) -> dict[str, object]:
        now = _utc_now()
        score_id = f"clawscore-{uuid4().hex}"
        with self.connect() as conn:
            conn.execute("DELETE FROM claw_scores WHERE attempt_id = %s", (attempt_id,))
            row = conn.execute(
                """
                INSERT INTO claw_scores
                (score_id, attempt_id, metrics, total_score, passed, failure_reasons, created_at)
                VALUES (%s, %s, %s::jsonb, %s, %s, %s::jsonb, %s)
                RETURNING *
                """,
                (score_id, attempt_id, _json_dump(metrics), total_score, passed, _json_dump(failure_reasons or []), now),
            ).fetchone()
            conn.commit()
        assert row is not None
        return self._claw_score_from_row(dict(row))

    def get_claw_run(self, run_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            run_row = conn.execute("SELECT * FROM claw_runs WHERE run_id = %s", (run_id,)).fetchone()
            if run_row is None:
                return None
            attempt_rows = conn.execute(
                "SELECT * FROM claw_attempts WHERE run_id = %s ORDER BY attempt_index ASC",
                (run_id,),
            ).fetchall()
            attempts: list[dict[str, object]] = []
            for attempt_row in attempt_rows:
                attempt = self._claw_attempt_from_row(dict(attempt_row))
                traj_rows = conn.execute(
                    "SELECT * FROM claw_trajectories WHERE attempt_id = %s ORDER BY event_index ASC",
                    (attempt["attempt_id"],),
                ).fetchall()
                score_row = conn.execute("SELECT * FROM claw_scores WHERE attempt_id = %s", (attempt["attempt_id"],)).fetchone()
                attempt["trajectory"] = [self._claw_trajectory_from_row(dict(row)) for row in traj_rows]
                attempt["score"] = self._claw_score_from_row(dict(score_row)) if score_row else None
                attempts.append(attempt)
        run = self._claw_run_from_row(dict(run_row))
        run["attempts"] = attempts
        return run

    def list_claw_runs(self, task_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if task_id is None:
                rows = conn.execute("SELECT * FROM claw_runs ORDER BY created_at DESC LIMIT %s", (limit,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM claw_runs WHERE task_id = %s ORDER BY created_at DESC LIMIT %s",
                    (task_id, limit),
                ).fetchall()
        return [self._claw_run_from_row(dict(row)) for row in rows]

    def list_claw_leaderboard(self, limit: int = 100) -> list[dict[str, object]]:
        runs = self.list_claw_runs(limit=1000)
        grouped: dict[str, list[dict[str, object]]] = {}
        for run in runs:
            grouped.setdefault(str(run["task_id"]), []).append(run)
        rows: list[dict[str, object]] = []
        for task_id, task_runs in grouped.items():
            completed = [run for run in task_runs if run["status"] in {"succeeded", "failed"}]
            if not completed:
                continue
            latest = completed[0]
            summary = latest.get("summary") if isinstance(latest.get("summary"), dict) else {}
            rows.append(
                {
                    "task_id": task_id,
                    "latest_run_id": latest["run_id"],
                    "latest_status": latest["status"],
                    "run_count": len(completed),
                    "pass_rate": summary.get("pass_rate", 0),
                    "average_score": summary.get("average_score", 0),
                    "best_score": summary.get("best_score", 0),
                    "updated_at": latest["updated_at"],
                }
            )
        return rows[:limit]
