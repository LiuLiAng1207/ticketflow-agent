from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from uuid import uuid4

from .models import AttachmentEvidence, AttachmentRecord, CustomerProfile, ExternalOpRecord, OrderRecord, TicketRecord


def _query_terms(query: str) -> list[str]:
    lowered = query.lower()
    terms: list[str] = []

    for token in re.findall(r"[a-z0-9]{2,}", lowered):
        if token not in terms:
            terms.append(token)

    for block in re.findall("[一-鿿]{2,}", query):
        if block not in terms:
            terms.append(block)
        for size in (2, 3):
            for idx in range(max(len(block) - size + 1, 0)):
                gram = block[idx : idx + size]
                if gram not in terms:
                    terms.append(gram)
    return terms


SCHEMA = """
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
    amount REAL NOT NULL,
    eligible_for_refund INTEGER NOT NULL
);

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
    approval_required INTEGER NOT NULL,
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
    created_at TEXT NOT NULL,
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
    metadata TEXT NOT NULL DEFAULT '{}',
    parse_status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS attachment_evidence (
    evidence_id TEXT PRIMARY KEY,
    attachment_id TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    evidence_type TEXT NOT NULL,
    extracted_text TEXT NOT NULL DEFAULT '',
    visual_summary TEXT NOT NULL DEFAULT '',
    entities TEXT NOT NULL DEFAULT '{}',
    confidence REAL NOT NULL DEFAULT 0,
    source_span TEXT,
    bbox TEXT,
    risk_flags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS escalations (
    escalation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    reason TEXT NOT NULL,
    priority TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS refund_requests (
    refund_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    order_id TEXT NOT NULL,
    amount REAL,
    rationale TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    payload TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS external_email_deliveries (
    delivery_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    message_type TEXT NOT NULL,
    recipient TEXT NOT NULL,
    subject TEXT NOT NULL,
    status TEXT NOT NULL,
    provider_message_id TEXT,
    latency_ms INTEGER,
    error_message TEXT,
    payload TEXT NOT NULL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS workflow_tasks (
    task_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    thread_id TEXT,
    status TEXT NOT NULL,
    mode TEXT NOT NULL,
    celery_task_id TEXT,
    result_json TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    cancelled_at TEXT
);

CREATE TABLE IF NOT EXISTS outbox_events (
    event_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    business_key TEXT NOT NULL,
    status TEXT NOT NULL,
    payload TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    last_error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    delivered_at TEXT,
    UNIQUE(ticket_id, operation_type, business_key)
);

CREATE TABLE IF NOT EXISTS idempotency_keys (
    key_scope TEXT NOT NULL,
    key_value TEXT NOT NULL,
    request_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    response_payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(key_scope, key_value)
);

CREATE TABLE IF NOT EXISTS external_operation_locks (
    lock_key TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    operation_type TEXT NOT NULL,
    acquired_by TEXT NOT NULL,
    expires_at TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS approval_requests (
    approval_id TEXT PRIMARY KEY,
    ticket_id TEXT NOT NULL,
    thread_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    tool_args TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    requested_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    expires_at TEXT
);

CREATE TABLE IF NOT EXISTS approval_decisions (
    decision_id TEXT PRIMARY KEY,
    approval_id TEXT NOT NULL,
    decision TEXT NOT NULL,
    reviewer TEXT NOT NULL,
    comment TEXT,
    edited_action TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS agent_skills (
    skill_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    risk_level TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    approval_required INTEGER NOT NULL,
    current_version TEXT NOT NULL,
    manifest TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS skill_versions (
    skill_id TEXT NOT NULL,
    version TEXT NOT NULL,
    manifest TEXT NOT NULL,
    skill_doc TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(skill_id, version)
);

CREATE TABLE IF NOT EXISTS skill_permissions (
    skill_id TEXT NOT NULL,
    permission TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY(skill_id, permission)
);

CREATE TABLE IF NOT EXISTS skill_runs (
    run_id TEXT PRIMARY KEY,
    skill_id TEXT NOT NULL,
    version TEXT,
    actor TEXT NOT NULL,
    ticket_id TEXT,
    status TEXT NOT NULL,
    input_payload TEXT NOT NULL,
    result_payload TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    requires_approval INTEGER NOT NULL DEFAULT 0,
    idempotency_key TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS claw_tasks (
    task_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    goal TEXT NOT NULL,
    category TEXT NOT NULL,
    enabled INTEGER NOT NULL,
    pass_k INTEGER NOT NULL,
    manifest TEXT NOT NULL,
    task_doc TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claw_runs (
    run_id TEXT PRIMARY KEY,
    task_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    status TEXT NOT NULL,
    pass_k INTEGER NOT NULL,
    config_payload TEXT NOT NULL DEFAULT '{}',
    summary_payload TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS claw_attempts (
    attempt_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    attempt_index INTEGER NOT NULL,
    status TEXT NOT NULL,
    input_payload TEXT NOT NULL DEFAULT '{}',
    output_payload TEXT NOT NULL DEFAULT '{}',
    error_message TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    completed_at TEXT,
    UNIQUE(run_id, attempt_index)
);

CREATE TABLE IF NOT EXISTS claw_trajectories (
    trajectory_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    event_index INTEGER NOT NULL,
    event_type TEXT NOT NULL,
    detail TEXT NOT NULL,
    payload TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS claw_scores (
    score_id TEXT PRIMARY KEY,
    attempt_id TEXT NOT NULL,
    metrics TEXT NOT NULL,
    total_score REAL NOT NULL,
    passed INTEGER NOT NULL,
    failure_reasons TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS observability_events (
    event_id TEXT PRIMARY KEY,
    trace_id TEXT NOT NULL,
    span_name TEXT NOT NULL,
    component TEXT NOT NULL,
    status TEXT NOT NULL,
    latency_ms INTEGER NOT NULL DEFAULT 0,
    error_type TEXT,
    payload_summary TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
);
"""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(payload: dict[str, object] | list[object] | None) -> str:
    return json.dumps(payload or {}, ensure_ascii=False)


def _json_load(raw: str | None) -> object:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"raw": raw}


class TicketFlowRepository:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        try:
            yield connection
        finally:
            connection.close()

    def init_database(self) -> None:
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            self._ensure_schema_migrations(conn)
            conn.commit()

    @staticmethod
    def _ensure_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
        existing = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
        for column_name, column_type in columns.items():
            if column_name not in existing:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {column_name} {column_type}")

    def _ensure_schema_migrations(self, conn: sqlite3.Connection) -> None:
        self._ensure_columns(
            conn,
            "tickets",
            {
                "source_dataset": "TEXT",
                "source_ticket_ref": "TEXT",
                "source_language": "TEXT",
                "source_queue": "TEXT",
                "source_subject": "TEXT",
                "source_body": "TEXT",
            },
        )

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
                    reader = csv.DictReader(handle)
                    rows = list(reader)
                if not rows:
                    continue
                conn.execute(f"DELETE FROM {table}")
                columns = rows[0].keys()
                placeholders = ", ".join(["?"] * len(columns))
                conn.executemany(
                    f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({placeholders})",
                    [tuple(self._coerce_value(value) for value in row.values()) for row in rows],
                )
            conn.commit()

    @staticmethod
    def _coerce_value(value: str) -> object:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return 1 if lowered == "true" else 0
        return value

    def bootstrap(self, seed_dir: Path) -> None:
        self.init_database()
        with self.connect() as conn:
            count = conn.execute("SELECT COUNT(*) FROM tickets").fetchone()[0]
        if count == 0:
            self.load_seed_directory(seed_dir)

    def reset_from_seed(self, seed_dir: Path) -> None:
        self.load_seed_directory(seed_dir)
        with self.connect() as conn:
            conn.execute("DELETE FROM attachment_evidence")
            conn.execute("DELETE FROM escalations")
            conn.execute("DELETE FROM refund_requests")
            conn.execute("DELETE FROM audit_log")
            conn.execute("DELETE FROM external_email_deliveries")
            conn.execute("DELETE FROM observability_events")
            conn.execute("DELETE FROM skill_runs")
            conn.commit()

    def list_open_tickets(self, limit: int = 50) -> list[TicketRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM tickets WHERE status = 'open' ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [TicketRecord.model_validate(dict(row)) for row in rows]

    def get_ticket(self, ticket_id: str) -> TicketRecord:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM tickets WHERE ticket_id = ?", (ticket_id,)).fetchone()
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, 'agent_chat', ?, 'zh', 'agent_chat', ?, ?)
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
            row = conn.execute("SELECT * FROM customers WHERE customer_id = ?", (customer_id,)).fetchone()
        if row is None:
            return None
        return CustomerProfile.model_validate(dict(row))

    def get_order_status(self, order_id: str | None) -> OrderRecord | None:
        if not order_id:
            return None
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM orders WHERE order_id = ?", (order_id,)).fetchone()
        if row is None:
            return None
        payload = dict(row)
        payload["eligible_for_refund"] = bool(payload["eligible_for_refund"])
        return OrderRecord.model_validate(payload)

    def get_ticket_history(self, ticket_id: str, limit: int = 5) -> list[dict[str, str]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT event_id, ticket_id, message, created_at, agent_name FROM ticket_history WHERE ticket_id = ? ORDER BY created_at DESC LIMIT ?",
                (ticket_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

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
                    (attachment_id, ticket_id, filename, file_type, source_dataset, storage_path, content_hash, ocr_text, visual_summary, metadata)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(metadata, ensure_ascii=False),
                ),
            )
            conn.commit()
        return self.get_ticket_attachment(attachment_id)

    def get_ticket_attachment(self, attachment_id: str) -> AttachmentRecord:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM ticket_attachments WHERE attachment_id = ?", (attachment_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown attachment_id: {attachment_id}")
        return self._attachment_from_row(row)

    def list_ticket_attachments(self, ticket_id: str) -> list[AttachmentRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM ticket_attachments WHERE ticket_id = ? ORDER BY created_at ASC, attachment_id ASC",
                (ticket_id,),
            ).fetchall()
        return [self._attachment_from_row(row) for row in rows]

    def replace_attachment_evidence(self, ticket_id: str, evidence: list[AttachmentEvidence]) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM attachment_evidence WHERE ticket_id = ?", (ticket_id,))
            for item in evidence:
                conn.execute(
                    """
                    INSERT INTO attachment_evidence
                        (evidence_id, attachment_id, ticket_id, evidence_type, extracted_text, visual_summary, entities, confidence, source_span, bbox, risk_flags, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        item.evidence_id,
                        item.attachment_id,
                        item.ticket_id,
                        item.evidence_type,
                        item.extracted_text,
                        item.visual_summary,
                        json.dumps(item.entities, ensure_ascii=False),
                        item.confidence,
                        item.source_span,
                        json.dumps(item.bbox, ensure_ascii=False) if item.bbox is not None else None,
                        json.dumps(item.risk_flags, ensure_ascii=False),
                        json.dumps(item.metadata, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def update_attachment_parse_status(self, attachment_id: str, status: str) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE ticket_attachments SET parse_status = ? WHERE attachment_id = ?", (status, attachment_id))
            conn.commit()

    def list_attachment_evidence(self, ticket_id: str) -> list[AttachmentEvidence]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM attachment_evidence WHERE ticket_id = ? ORDER BY evidence_id ASC",
                (ticket_id,),
            ).fetchall()
        return [self._attachment_evidence_from_row(row) for row in rows]

    @staticmethod
    def _attachment_from_row(row: sqlite3.Row) -> AttachmentRecord:
        payload = dict(row)
        payload["metadata"] = json.loads(payload["metadata"] or "{}")
        return AttachmentRecord.model_validate(payload)

    @staticmethod
    def _attachment_evidence_from_row(row: sqlite3.Row) -> AttachmentEvidence:
        payload = dict(row)
        payload["entities"] = json.loads(payload["entities"] or "{}")
        payload["bbox"] = json.loads(payload["bbox"]) if payload.get("bbox") else None
        payload["risk_flags"] = json.loads(payload["risk_flags"] or "[]")
        payload["metadata"] = json.loads(payload["metadata"] or "{}")
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
            rows = conn.execute("SELECT event_id, ticket_id, message, created_at, agent_name FROM ticket_history").fetchall()
        return [dict(row) for row in rows]

    def search_kb(self, query: str, product: str, limit: int = 5) -> list[dict[str, object]]:
        query_terms = _query_terms(query)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM kb_articles WHERE product = ? OR category LIKE ?",
                (product, f"%{product.split()[0].lower()}%"),
            ).fetchall()
        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            text = f"{row['title']} {row['body']} {row['tags']}".lower()
            score = sum(1 for term in query_terms if term in text)
            if row["product"] == product:
                score += 1.5
            if score:
                scored_rows.append((score, row))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [dict(row) | {"score": score} for score, row in scored_rows[:limit]]

    def search_policy_text(self, query: str, limit: int = 5) -> list[dict[str, object]]:
        terms = _query_terms(query)
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM policies").fetchall()
        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            text = f"{row['title']} {row['body']} {row['action_type']}".lower()
            score = sum(1 for term in terms if term in text)
            if row["approval_required"]:
                score += 0.2
            if score:
                scored_rows.append((score, row))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [dict(row) | {"score": score} for score, row in scored_rows[:limit]]

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
        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            text = " ".join(
                str(part or "")
                for part in (
                    row["message"],
                    row["agent_name"],
                    row["title"],
                    row["body"],
                    row["product"],
                    row["expected_category"],
                )
            ).lower()
            score = sum(1 for term in terms if term in text)
            if row["expected_category"] and any(term in str(row["expected_category"]).lower() for term in terms):
                score += 1.0
            if row["product"] and any(term in str(row["product"]).lower() for term in terms):
                score += 0.8
            if row["title"] and any(term in str(row["title"]).lower() for term in terms):
                score += 0.5
            if score:
                scored_rows.append((score, row))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [dict(row) | {"score": score} for score, row in scored_rows[:limit]]

    def lookup_policy(self, action_type: str, query: str = "", limit: int = 5) -> list[dict[str, object]]:
        terms = _query_terms(query)
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM policies WHERE action_type = ? OR body LIKE ?",
                (action_type, f"%{action_type}%"),
            ).fetchall()
        scored_rows: list[tuple[float, sqlite3.Row]] = []
        for row in rows:
            text = f"{row['title']} {row['body']} {row['action_type']}".lower()
            score = 1.0 if row["action_type"] == action_type else 0.0
            score += sum(1 for term in terms if term in text)
            scored_rows.append((score, row))
        scored_rows.sort(key=lambda item: item[0], reverse=True)
        return [dict(row) | {"score": score} for score, row in scored_rows[:limit]]

    def get_reply_templates(self, category: str, limit: int = 3) -> list[str]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT body FROM reply_templates WHERE category = ? LIMIT ?",
                (category, limit),
            ).fetchall()
        return [row["body"] for row in rows]

    def create_escalation(self, ticket_id: str, reason: str, priority: str) -> dict[str, object]:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO escalations (ticket_id, reason, priority) VALUES (?, ?, ?)",
                (ticket_id, reason, priority),
            )
            conn.commit()
        return {"escalation_id": cursor.lastrowid, "ticket_id": ticket_id, "priority": priority}

    def issue_refund_request(self, ticket_id: str, order_id: str, amount: float, rationale: str) -> dict[str, object]:
        with self.connect() as conn:
            cursor = conn.execute(
                "INSERT INTO refund_requests (ticket_id, order_id, amount, rationale) VALUES (?, ?, ?, ?)",
                (ticket_id, order_id, amount, rationale),
            )
            conn.commit()
        return {"refund_id": cursor.lastrowid, "ticket_id": ticket_id, "order_id": order_id, "amount": amount}

    def update_ticket_status(self, ticket_id: str, status: str) -> dict[str, object]:
        with self.connect() as conn:
            conn.execute("UPDATE tickets SET status = ? WHERE ticket_id = ?", (status, ticket_id))
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
            cursor = conn.execute(
                "INSERT INTO audit_log (ticket_id, actor, event_type, detail, payload) VALUES (?, ?, ?, ?, ?)",
                (ticket_id, actor, event_type, detail, json.dumps(payload, ensure_ascii=False)),
            )
            conn.commit()
        return {"audit_id": cursor.lastrowid}

    def list_audit_log(self, ticket_id: str, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT audit_id, ticket_id, actor, event_type, detail, payload, created_at
                FROM audit_log
                WHERE ticket_id = ?
                ORDER BY audit_id ASC
                LIMIT ?
                """,
                (ticket_id, limit),
            ).fetchall()
        events: list[dict[str, object]] = []
        for row in rows:
            payload_raw = row["payload"]
            try:
                payload = json.loads(payload_raw) if payload_raw else {}
            except json.JSONDecodeError:
                payload = {"raw": payload_raw}
            events.append(
                {
                    "audit_id": row["audit_id"],
                    "ticket_id": row["ticket_id"],
                    "actor": row["actor"],
                    "event_type": row["event_type"],
                    "detail": row["detail"],
                    "payload": payload,
                    "created_at": row["created_at"],
                }
            )
        return events

    @staticmethod
    def _observability_event_from_row(row: sqlite3.Row) -> dict[str, object]:
        payload_raw = row["payload_summary"]
        try:
            payload_summary = json.loads(payload_raw) if payload_raw else {}
        except json.JSONDecodeError:
            payload_summary = {"raw": payload_raw}
        return {
            "event_id": row["event_id"],
            "trace_id": row["trace_id"],
            "span_name": row["span_name"],
            "component": row["component"],
            "status": row["status"],
            "latency_ms": row["latency_ms"],
            "error_type": row["error_type"],
            "payload_summary": payload_summary,
            "created_at": row["created_at"],
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
            conn.execute(
                """
                INSERT INTO observability_events
                (event_id, trace_id, span_name, component, status, latency_ms, error_type, payload_summary, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_id,
                    trace_id,
                    span_name,
                    component,
                    status,
                    int(latency_ms),
                    error_type,
                    json.dumps(payload_summary or {}, ensure_ascii=False),
                    now,
                ),
            )
            conn.commit()
        rows = self.list_observability_events(trace_id=trace_id, limit=1)
        return rows[0] if rows else {"event_id": event_id, "trace_id": trace_id}

    def list_observability_events(self, trace_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if trace_id is None:
                rows = conn.execute(
                    "SELECT * FROM observability_events ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM observability_events WHERE trace_id = ? ORDER BY created_at ASC LIMIT ?",
                    (trace_id, limit),
                ).fetchall()
        return [self._observability_event_from_row(row) for row in rows]

    @staticmethod
    def _workflow_task_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "task_id": row["task_id"],
            "ticket_id": row["ticket_id"],
            "thread_id": row["thread_id"],
            "status": row["status"],
            "mode": row["mode"],
            "celery_task_id": row["celery_task_id"],
            "result": _json_load(row["result_json"]),
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "completed_at": row["completed_at"],
            "cancelled_at": row["cancelled_at"],
        }

    def create_workflow_task(self, ticket_id: str, mode: str, thread_id: str | None = None) -> dict[str, object]:
        task_id = f"task-{uuid4().hex}"
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO workflow_tasks
                (task_id, ticket_id, thread_id, status, mode, created_at, updated_at)
                VALUES (?, ?, ?, 'queued', ?, ?, ?)
                """,
                (task_id, ticket_id, thread_id, mode, now, now),
            )
            conn.commit()
        task = self.get_workflow_task(task_id)
        assert task is not None
        return task

    def get_workflow_task(self, task_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM workflow_tasks WHERE task_id = ?", (task_id,)).fetchone()
        if row is None:
            return None
        return self._workflow_task_from_row(row)

    def get_workflow_task_by_thread(self, thread_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute(
                "SELECT * FROM workflow_tasks WHERE thread_id = ? ORDER BY created_at DESC LIMIT 1",
                (thread_id,),
            ).fetchone()
        if row is None:
            return None
        return self._workflow_task_from_row(row)

    def list_workflow_tasks(self, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM workflow_tasks ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._workflow_task_from_row(row) for row in rows]

    def mark_workflow_task_running(self, task_id: str, celery_task_id: str | None = None) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'running', celery_task_id = COALESCE(?, celery_task_id),
                    started_at = COALESCE(started_at, ?), updated_at = ?
                WHERE task_id = ?
                """,
                (celery_task_id, now, now, task_id),
            )
            conn.commit()
        task = self.get_workflow_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return task

    def set_workflow_task_celery_id(self, task_id: str, celery_task_id: str) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                "UPDATE workflow_tasks SET celery_task_id = ?, updated_at = ? WHERE task_id = ?",
                (celery_task_id, now, task_id),
            )
            conn.commit()
        task = self.get_workflow_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return task

    def complete_workflow_task(self, task_id: str, result: dict[str, object]) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'succeeded', result_json = ?, error_message = NULL,
                    completed_at = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (_json_dump(result), now, now, task_id),
            )
            conn.commit()
        task = self.get_workflow_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return task

    def wait_workflow_task_for_approval(self, task_id: str, thread_id: str, approval_id: str | None) -> dict[str, object]:
        now = _utc_now()
        result = {"interrupted": True, "approval_id": approval_id, "thread_id": thread_id}
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'waiting_approval', thread_id = ?, result_json = ?,
                    error_message = NULL, updated_at = ?
                WHERE task_id = ?
                """,
                (thread_id, _json_dump(result), now, task_id),
            )
            conn.commit()
        task = self.get_workflow_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        task["approval_id"] = approval_id
        return task

    def fail_workflow_task(self, task_id: str, error_message: str) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'failed', error_message = ?, completed_at = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (error_message, now, now, task_id),
            )
            conn.commit()
        task = self.get_workflow_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return task

    def cancel_workflow_task(self, task_id: str, reason: str = "cancelled") -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE workflow_tasks
                SET status = 'cancelled', error_message = ?, cancelled_at = ?, updated_at = ?
                WHERE task_id = ?
                """,
                (reason, now, now, task_id),
            )
            conn.commit()
        task = self.get_workflow_task(task_id)
        if task is None:
            raise KeyError(f"Unknown task_id: {task_id}")
        return task

    @staticmethod
    def _outbox_event_from_row(row: sqlite3.Row, *, deduplicated: bool = False) -> dict[str, object]:
        return {
            "event_id": row["event_id"],
            "ticket_id": row["ticket_id"],
            "operation_type": row["operation_type"],
            "business_key": row["business_key"],
            "status": row["status"],
            "payload": _json_load(row["payload"]),
            "attempts": row["attempts"],
            "last_error": row["last_error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "delivered_at": row["delivered_at"],
            "deduplicated": deduplicated,
        }

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
            try:
                conn.execute(
                    """
                    INSERT INTO outbox_events
                    (event_id, ticket_id, operation_type, business_key, status, payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?, 'pending', ?, ?, ?)
                    """,
                    (event_id, ticket_id, operation_type, business_key, _json_dump(payload), now, now),
                )
                conn.commit()
                deduplicated = False
            except sqlite3.IntegrityError:
                deduplicated = True
            row = conn.execute(
                """
                SELECT * FROM outbox_events
                WHERE ticket_id = ? AND operation_type = ? AND business_key = ?
                """,
                (ticket_id, operation_type, business_key),
            ).fetchone()
        assert row is not None
        return self._outbox_event_from_row(row, deduplicated=deduplicated)

    def get_outbox_event(self, event_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM outbox_events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            return None
        return self._outbox_event_from_row(row)

    def list_outbox_events(self, status: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM outbox_events ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM outbox_events WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [self._outbox_event_from_row(row) for row in rows]

    def mark_outbox_event_delivered(self, event_id: str) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE outbox_events
                SET status = 'delivered', delivered_at = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (now, now, event_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM outbox_events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        return self._outbox_event_from_row(row)

    def mark_outbox_event_failed(self, event_id: str, error_message: str) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                """
                UPDATE outbox_events
                SET status = 'failed', attempts = attempts + 1, last_error = ?, updated_at = ?
                WHERE event_id = ?
                """,
                (error_message, now, event_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM outbox_events WHERE event_id = ?", (event_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown event_id: {event_id}")
        return self._outbox_event_from_row(row)

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
            try:
                conn.execute(
                    """
                    INSERT INTO idempotency_keys
                    (key_scope, key_value, request_hash, status, response_payload, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (key_scope, key_value, request_hash, status, _json_dump(response_payload), now, now),
                )
                conn.commit()
                deduplicated = False
            except sqlite3.IntegrityError:
                deduplicated = True
            row = conn.execute(
                "SELECT * FROM idempotency_keys WHERE key_scope = ? AND key_value = ?",
                (key_scope, key_value),
            ).fetchone()
        assert row is not None
        return {
            "key_scope": row["key_scope"],
            "key_value": row["key_value"],
            "request_hash": row["request_hash"],
            "status": row["status"],
            "response_payload": _json_load(row["response_payload"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "deduplicated": deduplicated,
        }

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
            try:
                conn.execute(
                    """
                    INSERT INTO external_operation_locks
                    (lock_key, ticket_id, operation_type, acquired_by, expires_at, created_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (lock_key, ticket_id, operation_type, acquired_by, expires_at, now),
                )
                conn.commit()
                acquired = True
            except sqlite3.IntegrityError:
                acquired = False
            row = conn.execute(
                "SELECT * FROM external_operation_locks WHERE lock_key = ?",
                (lock_key,),
            ).fetchone()
        assert row is not None
        return {
            "lock_key": row["lock_key"],
            "ticket_id": row["ticket_id"],
            "operation_type": row["operation_type"],
            "acquired_by": row["acquired_by"],
            "expires_at": row["expires_at"],
            "created_at": row["created_at"],
            "acquired": acquired,
        }

    @staticmethod
    def _approval_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "approval_id": row["approval_id"],
            "ticket_id": row["ticket_id"],
            "thread_id": row["thread_id"],
            "tool_name": row["tool_name"],
            "tool_args": _json_load(row["tool_args"]),
            "payload": _json_load(row["payload"]),
            "status": row["status"],
            "requested_by": row["requested_by"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "expires_at": row["expires_at"],
        }

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
            conn.execute(
                """
                INSERT INTO approval_requests
                (approval_id, ticket_id, thread_id, tool_name, tool_args, payload, status, requested_by, created_at, updated_at, expires_at)
                VALUES (?, ?, ?, ?, ?, ?, 'pending', ?, ?, ?, ?)
                """,
                (
                    approval_id,
                    ticket_id,
                    thread_id,
                    tool_name,
                    _json_dump(tool_args),
                    _json_dump(payload),
                    requested_by,
                    now,
                    now,
                    expires_at,
                ),
            )
            conn.commit()
        approval = self.get_approval_request(approval_id)
        assert approval is not None
        return approval

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
                WHERE thread_id = ? AND tool_name = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (thread_id, tool_name),
            ).fetchone()
        if row is not None:
            approval = self._approval_from_row(row)
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
            row = conn.execute("SELECT * FROM approval_requests WHERE approval_id = ?", (approval_id,)).fetchone()
        if row is None:
            return None
        return self._approval_from_row(row)

    def list_approval_requests(self, status: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if status is None:
                rows = conn.execute(
                    "SELECT * FROM approval_requests ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM approval_requests WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
        return [self._approval_from_row(row) for row in rows]

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
            existing = conn.execute("SELECT * FROM approval_requests WHERE approval_id = ?", (approval_id,)).fetchone()
            if existing is None:
                raise KeyError(f"Unknown approval_id: {approval_id}")
            conn.execute(
                """
                INSERT INTO approval_decisions
                (decision_id, approval_id, decision, reviewer, comment, edited_action, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (decision_id, approval_id, decision, reviewer, comment, _json_dump(edited_action), now),
            )
            conn.execute(
                "UPDATE approval_requests SET status = ?, updated_at = ? WHERE approval_id = ?",
                (request_status, now, approval_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM approval_decisions WHERE decision_id = ?", (decision_id,)).fetchone()
        assert row is not None
        return {
            "decision_id": row["decision_id"],
            "approval_id": row["approval_id"],
            "decision": row["decision"],
            "reviewer": row["reviewer"],
            "comment": row["comment"],
            "edited_action": _json_load(row["edited_action"]),
            "created_at": row["created_at"],
        }

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
            cursor = conn.execute(
                """
                INSERT INTO external_email_deliveries
                    (ticket_id, message_type, recipient, subject, status, provider_message_id, latency_ms, error_message, payload)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    json.dumps(payload, ensure_ascii=False),
                ),
            )
            conn.commit()
        return {"delivery_id": cursor.lastrowid}

    def list_external_email_deliveries(self, ticket_id: str) -> list[ExternalOpRecord]:
        with self.connect() as conn:
            rows = conn.execute(
                """
                SELECT delivery_id, message_type, recipient, subject, status, provider_message_id, latency_ms, error_message, payload, created_at
                FROM external_email_deliveries
                WHERE ticket_id = ?
                ORDER BY delivery_id ASC
                """,
                (ticket_id,),
            ).fetchall()
        records: list[ExternalOpRecord] = []
        for row in rows:
            payload = json.loads(row["payload"]) if row["payload"] else {}
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
                    payload=payload,
                )
            )
        return records

    @staticmethod
    def _skill_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "skill_id": row["skill_id"],
            "name": row["name"],
            "description": row["description"],
            "risk_level": row["risk_level"],
            "enabled": bool(row["enabled"]),
            "approval_required": bool(row["approval_required"]),
            "current_version": row["current_version"],
            "manifest": _json_load(row["manifest"]),
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _skill_run_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "run_id": row["run_id"],
            "skill_id": row["skill_id"],
            "version": row["version"],
            "actor": row["actor"],
            "ticket_id": row["ticket_id"],
            "status": row["status"],
            "input_payload": _json_load(row["input_payload"]),
            "result_payload": _json_load(row["result_payload"]),
            "error_message": row["error_message"],
            "requires_approval": bool(row["requires_approval"]),
            "idempotency_key": row["idempotency_key"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }

    def upsert_agent_skill(self, manifest: dict[str, object], skill_doc: str = "") -> dict[str, object]:
        now = _utc_now()
        skill_id = str(manifest["skill_id"])
        version = str(manifest["version"])
        permissions = [str(item) for item in manifest.get("permissions", [])]
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO agent_skills
                (skill_id, name, description, risk_level, enabled, approval_required, current_version, manifest, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(skill_id) DO UPDATE SET
                    name = excluded.name,
                    description = excluded.description,
                    risk_level = excluded.risk_level,
                    approval_required = excluded.approval_required,
                    current_version = excluded.current_version,
                    manifest = excluded.manifest,
                    updated_at = excluded.updated_at
                """,
                (
                    skill_id,
                    str(manifest["name"]),
                    str(manifest["description"]),
                    str(manifest["risk_level"]),
                    1 if bool(manifest.get("enabled", True)) else 0,
                    1 if bool(manifest.get("approval_required", False)) else 0,
                    version,
                    _json_dump(manifest),
                    now,
                    now,
                ),
            )
            conn.execute(
                """
                INSERT INTO skill_versions (skill_id, version, manifest, skill_doc, created_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(skill_id, version) DO UPDATE SET
                    manifest = excluded.manifest,
                    skill_doc = excluded.skill_doc
                """,
                (skill_id, version, _json_dump(manifest), skill_doc, now),
            )
            conn.execute("DELETE FROM skill_permissions WHERE skill_id = ?", (skill_id,))
            conn.executemany(
                "INSERT INTO skill_permissions (skill_id, permission, created_at) VALUES (?, ?, ?)",
                [(skill_id, permission, now) for permission in permissions],
            )
            conn.commit()
            row = conn.execute("SELECT * FROM agent_skills WHERE skill_id = ?", (skill_id,)).fetchone()
        assert row is not None
        return self._skill_from_row(row)

    def list_agent_skills(self, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_skills ORDER BY skill_id ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._skill_from_row(row) for row in rows]

    def get_agent_skill(self, skill_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM agent_skills WHERE skill_id = ?", (skill_id,)).fetchone()
        return self._skill_from_row(row) if row else None

    def set_agent_skill_enabled(self, skill_id: str, enabled: bool) -> dict[str, object]:
        now = _utc_now()
        with self.connect() as conn:
            conn.execute(
                "UPDATE agent_skills SET enabled = ?, updated_at = ? WHERE skill_id = ?",
                (1 if enabled else 0, now, skill_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM agent_skills WHERE skill_id = ?", (skill_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown skill_id: {skill_id}")
        return self._skill_from_row(row)

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
            conn.execute(
                """
                INSERT INTO skill_runs
                (run_id, skill_id, version, actor, ticket_id, status, input_payload, result_payload, error_message,
                 requires_approval, idempotency_key, created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    1 if requires_approval else 0,
                    idempotency_key,
                    now,
                    now,
                    completed_at,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM skill_runs WHERE run_id = ?", (run_id,)).fetchone()
        assert row is not None
        return self._skill_run_from_row(row)

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
            conn.execute(
                """
                UPDATE skill_runs
                SET status = ?, result_payload = ?, error_message = ?, requires_approval = ?,
                    updated_at = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (
                    status,
                    _json_dump(result_payload),
                    error_message,
                    1 if requires_approval else 0,
                    now,
                    completed_at,
                    run_id,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM skill_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown run_id: {run_id}")
        return self._skill_run_from_row(row)

    def list_skill_runs(self, skill_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if skill_id is None:
                rows = conn.execute(
                    "SELECT * FROM skill_runs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM skill_runs WHERE skill_id = ? ORDER BY created_at DESC LIMIT ?",
                    (skill_id, limit),
                ).fetchall()
        return [self._skill_run_from_row(row) for row in rows]

    @staticmethod
    def _claw_task_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "task_id": row["task_id"],
            "name": row["name"],
            "goal": row["goal"],
            "category": row["category"],
            "enabled": bool(row["enabled"]),
            "pass_k": row["pass_k"],
            "manifest": _json_load(row["manifest"]),
            "task_doc": row["task_doc"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    @staticmethod
    def _claw_run_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "run_id": row["run_id"],
            "task_id": row["task_id"],
            "actor": row["actor"],
            "status": row["status"],
            "pass_k": row["pass_k"],
            "config_payload": _json_load(row["config_payload"]),
            "summary": _json_load(row["summary_payload"]),
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }

    @staticmethod
    def _claw_attempt_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "attempt_id": row["attempt_id"],
            "run_id": row["run_id"],
            "attempt_index": row["attempt_index"],
            "status": row["status"],
            "input_payload": _json_load(row["input_payload"]),
            "output_payload": _json_load(row["output_payload"]),
            "error_message": row["error_message"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "completed_at": row["completed_at"],
        }

    @staticmethod
    def _claw_trajectory_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "trajectory_id": row["trajectory_id"],
            "attempt_id": row["attempt_id"],
            "event_index": row["event_index"],
            "event_type": row["event_type"],
            "detail": row["detail"],
            "payload": _json_load(row["payload"]),
            "created_at": row["created_at"],
        }

    @staticmethod
    def _claw_score_from_row(row: sqlite3.Row) -> dict[str, object]:
        return {
            "score_id": row["score_id"],
            "attempt_id": row["attempt_id"],
            "metrics": _json_load(row["metrics"]),
            "total_score": row["total_score"],
            "passed": bool(row["passed"]),
            "failure_reasons": _json_load(row["failure_reasons"]),
            "created_at": row["created_at"],
        }

    def upsert_claw_task(self, manifest: dict[str, object], task_doc: str = "") -> dict[str, object]:
        now = _utc_now()
        task_id = str(manifest["task_id"])
        with self.connect() as conn:
            conn.execute(
                """
                INSERT INTO claw_tasks
                (task_id, name, goal, category, enabled, pass_k, manifest, task_doc, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(task_id) DO UPDATE SET
                    name = excluded.name,
                    goal = excluded.goal,
                    category = excluded.category,
                    enabled = excluded.enabled,
                    pass_k = excluded.pass_k,
                    manifest = excluded.manifest,
                    task_doc = excluded.task_doc,
                    updated_at = excluded.updated_at
                """,
                (
                    task_id,
                    str(manifest["name"]),
                    str(manifest["goal"]),
                    str(manifest.get("category") or "ticketflow"),
                    1 if bool(manifest.get("enabled", True)) else 0,
                    int(manifest.get("pass_k") or 3),
                    _json_dump(manifest),
                    task_doc,
                    now,
                    now,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM claw_tasks WHERE task_id = ?", (task_id,)).fetchone()
        assert row is not None
        return self._claw_task_from_row(row)

    def list_claw_tasks(self, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM claw_tasks ORDER BY task_id ASC LIMIT ?", (limit,)).fetchall()
        return [self._claw_task_from_row(row) for row in rows]

    def get_claw_task(self, task_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM claw_tasks WHERE task_id = ?", (task_id,)).fetchone()
        return self._claw_task_from_row(row) if row else None

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
            conn.execute(
                """
                INSERT INTO claw_runs
                (run_id, task_id, actor, status, pass_k, config_payload, summary_payload, error_message,
                 created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    task_id,
                    actor,
                    status,
                    pass_k,
                    _json_dump(config_payload),
                    _json_dump({}),
                    None,
                    now,
                    now,
                    completed_at,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM claw_runs WHERE run_id = ?", (run_id,)).fetchone()
        assert row is not None
        return self._claw_run_from_row(row)

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
            conn.execute(
                """
                UPDATE claw_runs
                SET status = ?, summary_payload = ?, error_message = ?, updated_at = ?, completed_at = ?
                WHERE run_id = ?
                """,
                (status, _json_dump(summary_payload), error_message, now, completed_at, run_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM claw_runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown claw run_id: {run_id}")
        return self._claw_run_from_row(row)

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
            conn.execute(
                """
                INSERT INTO claw_attempts
                (attempt_id, run_id, attempt_index, status, input_payload, output_payload, error_message,
                 created_at, updated_at, completed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    attempt_id,
                    run_id,
                    attempt_index,
                    status,
                    _json_dump(input_payload),
                    _json_dump({}),
                    None,
                    now,
                    now,
                    completed_at,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM claw_attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
        assert row is not None
        return self._claw_attempt_from_row(row)

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
            conn.execute(
                """
                UPDATE claw_attempts
                SET status = ?, output_payload = ?, error_message = ?, updated_at = ?, completed_at = ?
                WHERE attempt_id = ?
                """,
                (status, _json_dump(output_payload), error_message, now, completed_at, attempt_id),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM claw_attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown claw attempt_id: {attempt_id}")
        return self._claw_attempt_from_row(row)

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
            row = conn.execute(
                "SELECT COALESCE(MAX(event_index), 0) + 1 AS event_index FROM claw_trajectories WHERE attempt_id = ?",
                (attempt_id,),
            ).fetchone()
            event_index = int(row["event_index"] if row else 1)
            conn.execute(
                """
                INSERT INTO claw_trajectories
                (trajectory_id, attempt_id, event_index, event_type, detail, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (trajectory_id, attempt_id, event_index, event_type, detail, _json_dump(payload), now),
            )
            conn.commit()
            stored = conn.execute("SELECT * FROM claw_trajectories WHERE trajectory_id = ?", (trajectory_id,)).fetchone()
        assert stored is not None
        return self._claw_trajectory_from_row(stored)

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
            conn.execute("DELETE FROM claw_scores WHERE attempt_id = ?", (attempt_id,))
            conn.execute(
                """
                INSERT INTO claw_scores
                (score_id, attempt_id, metrics, total_score, passed, failure_reasons, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    score_id,
                    attempt_id,
                    _json_dump(metrics),
                    total_score,
                    1 if passed else 0,
                    _json_dump(failure_reasons or []),
                    now,
                ),
            )
            conn.commit()
            row = conn.execute("SELECT * FROM claw_scores WHERE score_id = ?", (score_id,)).fetchone()
        assert row is not None
        return self._claw_score_from_row(row)

    def get_claw_run(self, run_id: str) -> dict[str, object] | None:
        with self.connect() as conn:
            run_row = conn.execute("SELECT * FROM claw_runs WHERE run_id = ?", (run_id,)).fetchone()
            if run_row is None:
                return None
            attempt_rows = conn.execute(
                "SELECT * FROM claw_attempts WHERE run_id = ? ORDER BY attempt_index ASC",
                (run_id,),
            ).fetchall()
            attempts: list[dict[str, object]] = []
            for attempt_row in attempt_rows:
                attempt = self._claw_attempt_from_row(attempt_row)
                traj_rows = conn.execute(
                    "SELECT * FROM claw_trajectories WHERE attempt_id = ? ORDER BY event_index ASC",
                    (attempt["attempt_id"],),
                ).fetchall()
                score_row = conn.execute(
                    "SELECT * FROM claw_scores WHERE attempt_id = ?",
                    (attempt["attempt_id"],),
                ).fetchone()
                attempt["trajectory"] = [self._claw_trajectory_from_row(row) for row in traj_rows]
                attempt["score"] = self._claw_score_from_row(score_row) if score_row else None
                attempts.append(attempt)
        run = self._claw_run_from_row(run_row)
        run["attempts"] = attempts
        return run

    def list_claw_runs(self, task_id: str | None = None, limit: int = 100) -> list[dict[str, object]]:
        with self.connect() as conn:
            if task_id is None:
                rows = conn.execute("SELECT * FROM claw_runs ORDER BY created_at DESC LIMIT ?", (limit,)).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM claw_runs WHERE task_id = ? ORDER BY created_at DESC LIMIT ?",
                    (task_id, limit),
                ).fetchall()
        return [self._claw_run_from_row(row) for row in rows]

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
