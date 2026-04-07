from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable

from .db import TicketFlowRepository
from .models import AuditEvent, Citation, ExternalOpRecord, RetrievedDoc


@dataclass(slots=True)
class TicketTools:
    repository: TicketFlowRepository
    max_retries: int = 1

    def search_kb(self, query: str, product: str, limit: int = 5) -> list[RetrievedDoc]:
        rows = self.repository.search_kb(query, product, limit)
        return [
            RetrievedDoc(
                doc_id=str(row["doc_id"]),
                source_type="kb",
                title=str(row["title"]),
                snippet=str(row["body"]),
                score=float(row.get("score", 0.0)),
                metadata={"product": row.get("product"), "category": row.get("category")},
                citations=[Citation(source_path=f"kb:{row['doc_id']}")],
            )
            for row in rows
        ]

    def get_customer_profile(self, customer_id: str) -> dict[str, Any] | None:
        profile = self.repository.get_customer_profile(customer_id)
        return profile.model_dump(mode="json") if profile is not None else None

    def get_order_status(self, order_id: str | None) -> dict[str, Any] | None:
        order = self.repository.get_order_status(order_id)
        return order.model_dump(mode="json") if order is not None else None

    def get_ticket_history(self, ticket_id: str, limit: int = 5) -> list[dict[str, Any]]:
        return self.repository.get_ticket_history(ticket_id, limit=limit)

    def lookup_policy(self, action_type: str, query: str = "", limit: int = 5) -> list[RetrievedDoc]:
        rows = self.repository.lookup_policy(action_type, query, limit)
        return [
            RetrievedDoc(
                doc_id=str(row["policy_id"]),
                source_type="policy",
                title=str(row["title"]),
                snippet=str(row["body"]),
                score=float(row.get("score", 0.0)),
                metadata={
                    "action_type": row.get("action_type"),
                    "approval_required": bool(row.get("approval_required")),
                    "priority_hint": row.get("priority_hint"),
                },
                citations=[Citation(source_path=f"policy:{row['policy_id']}")],
            )
            for row in rows
        ]

    def search_policy_text(self, query: str, limit: int = 5) -> list[RetrievedDoc]:
        rows = self.repository.search_policy_text(query, limit)
        return [
            RetrievedDoc(
                doc_id=str(row["policy_id"]),
                source_type="policy",
                title=str(row["title"]),
                snippet=str(row["body"]),
                score=float(row.get("score", 0.0)),
                metadata={
                    "action_type": row.get("action_type"),
                    "approval_required": bool(row.get("approval_required")),
                    "priority_hint": row.get("priority_hint"),
                },
                citations=[Citation(source_path=f"policy:{row['policy_id']}")],
            )
            for row in rows
        ]

    def create_escalation(self, ticket_id: str, reason: str, priority: str) -> dict[str, Any]:
        return self._run_with_retry(self.repository.create_escalation, ticket_id=ticket_id, reason=reason, priority=priority)

    def issue_refund_request(self, ticket_id: str, order_id: str, amount: float, rationale: str) -> dict[str, Any]:
        return self._run_with_retry(
            self.repository.issue_refund_request,
            ticket_id=ticket_id,
            order_id=order_id,
            amount=amount,
            rationale=rationale,
        )

    def update_ticket_status(self, ticket_id: str, status: str) -> dict[str, Any]:
        return self._run_with_retry(self.repository.update_ticket_status, ticket_id=ticket_id, status=status)

    def save_audit_log(self, ticket_id: str, actor: str, event_type: str, detail: str, payload: dict[str, Any]) -> dict[str, Any]:
        return self.repository.save_audit_log(ticket_id, actor, event_type, detail, payload)

    def save_external_email_delivery(self, ticket_id: str, record: ExternalOpRecord) -> dict[str, Any]:
        return self.repository.create_external_email_delivery(
            ticket_id=ticket_id,
            message_type=record.op_type,
            recipient=record.recipient,
            subject=record.subject,
            status=record.status,
            provider_message_id=record.provider_message_id,
            latency_ms=record.latency_ms,
            error_message=record.error_message,
            payload=record.payload,
        )

    def list_external_email_deliveries(self, ticket_id: str) -> list[ExternalOpRecord]:
        return self.repository.list_external_email_deliveries(ticket_id)

    def make_audit_event(self, actor: str, event_type: str, detail: str, payload: dict[str, Any] | None = None) -> AuditEvent:
        return AuditEvent(
            timestamp=datetime.now(),
            actor=actor,
            event_type=event_type,
            detail=detail,
            payload=payload or {},
        )

    def _run_with_retry(self, fn: Callable[..., dict[str, Any]], **kwargs: Any) -> dict[str, Any]:
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                return fn(**kwargs)
            except Exception as exc:  # pragma: no cover - simple retry wrapper
                last_error = exc
        assert last_error is not None
        raise last_error
