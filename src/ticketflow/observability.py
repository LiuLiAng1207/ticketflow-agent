from __future__ import annotations

import json
import time
from collections import defaultdict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterator
from uuid import uuid4


_CURRENT_TRACE_ID: ContextVar[str | None] = ContextVar("ticketflow_trace_id", default=None)
_METRICS: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
SECRET_TOKENS = ("secret", "password", "token", "authorization", "auth_code", "api_key", "apikey", "key")
MAX_STRING_LENGTH = 200


@dataclass(frozen=True, slots=True)
class TraceContext:
    trace_id: str
    actor: str = "system"


def generate_trace_id() -> str:
    return f"trace-{uuid4().hex}"


def get_trace_id() -> str:
    existing = _CURRENT_TRACE_ID.get()
    if existing:
        return existing
    trace_id = generate_trace_id()
    _CURRENT_TRACE_ID.set(trace_id)
    return trace_id


def set_trace_id(trace_id: str | None) -> str:
    selected = trace_id or generate_trace_id()
    _CURRENT_TRACE_ID.set(selected)
    return selected


def _is_secret_key(key: str) -> bool:
    lowered = key.lower()
    return any(token in lowered for token in SECRET_TOKENS)


def sanitize_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): "***redacted***" if _is_secret_key(str(key)) else sanitize_payload(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [sanitize_payload(item) for item in value[:50]]
    if isinstance(value, tuple):
        return [sanitize_payload(item) for item in value[:50]]
    if isinstance(value, str):
        if len(value) > MAX_STRING_LENGTH:
            return value[:MAX_STRING_LENGTH] + "..."
        return value
    return value


def classify_error(exc: BaseException | str | None, *, component: str = "api") -> str:
    component_map = {
        "api": "api_error",
        "worker": "worker_error",
        "rag": "rag_error",
        "llm": "llm_error",
        "tool": "tool_error",
        "approval": "approval_error",
        "outbox": "outbox_error",
        "mcp": "mcp_error",
    }
    normalized_component = component.strip().lower()
    if normalized_component in component_map:
        return component_map[normalized_component]
    message = str(exc or "").lower()
    if any(token in message for token in ("rag", "vector", "retriev", "embedding", "chroma", "qdrant")):
        return "rag_error"
    if any(token in message for token in ("llm", "jsondecode", "model", "openai", "deepseek", "minimind")):
        return "llm_error"
    if any(token in message for token in ("smtp", "outbox", "email", "delivery")):
        return "outbox_error"
    if any(token in message for token in ("approval", "review", "hitl")):
        return "approval_error"
    if any(token in message for token in ("mcp", "scope", "permission")):
        return "mcp_error"
    if "tool" in message:
        return "tool_error"
    return "unknown_error"


def json_log(logger: str, message: str, payload: dict[str, Any] | None = None) -> str:
    body = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "logger": logger,
        "message": message,
        **sanitize_payload(payload or {}),
    }
    return json.dumps(body, ensure_ascii=False, default=str)


def _labels_key(labels: dict[str, str] | None = None) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), str(value)) for key, value in (labels or {}).items()))


def record_metric(name: str, value: float = 1.0, labels: dict[str, str] | None = None) -> None:
    _METRICS[(name, _labels_key(labels))] += float(value)


def set_metric(name: str, value: float, labels: dict[str, str] | None = None) -> None:
    _METRICS[(name, _labels_key(labels))] = float(value)


def observe_latency(name: str, seconds: float, labels: dict[str, str] | None = None) -> None:
    record_metric(f"{name}_count", 1, labels)
    record_metric(f"{name}_sum", max(0.0, seconds), labels)


def render_prometheus_metrics() -> str:
    lines: list[str] = []
    for (name, labels), value in sorted(_METRICS.items(), key=lambda item: (item[0][0], item[0][1])):
        label_text = ""
        if labels:
            label_text = "{" + ",".join(f'{key}="{val}"' for key, val in labels) + "}"
        lines.append(f"{name}{label_text} {value}")
    if not lines:
        lines.append("ticketflow_metrics_bootstrap_total 0")
    return "\n".join(lines) + "\n"


def record_observability_event(
    repository: Any,
    *,
    trace_id: str | None = None,
    span_name: str,
    component: str,
    status: str,
    latency_ms: int = 0,
    error_type: str | None = None,
    payload_summary: dict[str, Any] | None = None,
) -> dict[str, object] | None:
    if not hasattr(repository, "record_observability_event"):
        return None
    return repository.record_observability_event(
        trace_id=trace_id or get_trace_id(),
        span_name=span_name,
        component=component,
        status=status,
        latency_ms=max(0, int(latency_ms)),
        error_type=error_type,
        payload_summary=sanitize_payload(payload_summary or {}),
    )


@contextmanager
def observed_operation(
    repository: Any,
    *,
    span_name: str,
    component: str,
    trace_id: str | None = None,
    payload_summary: dict[str, Any] | None = None,
) -> Iterator[TraceContext]:
    selected_trace_id = set_trace_id(trace_id or get_trace_id())
    started = time.perf_counter()
    try:
        yield TraceContext(trace_id=selected_trace_id)
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        error_type = classify_error(exc, component=component)
        record_metric("ticketflow_errors_total", 1, {"component": component, "error_type": error_type})
        record_observability_event(
            repository,
            trace_id=selected_trace_id,
            span_name=span_name,
            component=component,
            status="error",
            latency_ms=latency_ms,
            error_type=error_type,
            payload_summary={**(payload_summary or {}), "error_message_summary": str(exc)},
        )
        raise
    else:
        latency_ms = int((time.perf_counter() - started) * 1000)
        record_observability_event(
            repository,
            trace_id=selected_trace_id,
            span_name=span_name,
            component=component,
            status="ok",
            latency_ms=latency_ms,
            payload_summary=payload_summary or {},
        )
