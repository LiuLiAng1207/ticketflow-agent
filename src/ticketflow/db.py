from __future__ import annotations

import csv
import hashlib
import json
import re
import sqlite3
from contextlib import contextmanager
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
"""


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
