from __future__ import annotations

import json

from ticketflow.observability import classify_error, generate_trace_id, json_log, sanitize_payload


def test_trace_id_generation_is_stable_shape():
    trace_id = generate_trace_id()

    assert trace_id.startswith("trace-")
    assert len(trace_id) > 20


def test_error_classifier_covers_production_components():
    assert classify_error(RuntimeError("rag vector search timeout"), component="rag") == "rag_error"
    assert classify_error(RuntimeError("LLM JSONDecodeError"), component="llm") == "llm_error"
    assert classify_error(PermissionError("write tool disabled"), component="mcp") == "mcp_error"
    assert classify_error(RuntimeError("SMTP delivery failed"), component="outbox") == "outbox_error"
    assert classify_error(RuntimeError("approval missing"), component="approval") == "approval_error"
    assert classify_error(RuntimeError("tool failed"), component="tool") == "tool_error"


def test_json_log_sanitizes_secret_like_fields():
    payload = {
        "OPENAI_COMPAT_API_KEY": "sk-secret",
        "SMTP_AUTH_CODE": "smtp-secret",
        "nested": {"authorization": "Bearer token", "safe": "visible"},
    }

    line = json_log("ticketflow.test", "ok", payload)
    parsed = json.loads(line)

    assert parsed["OPENAI_COMPAT_API_KEY"] == "***redacted***"
    assert parsed["SMTP_AUTH_CODE"] == "***redacted***"
    assert parsed["nested"]["authorization"] == "***redacted***"
    assert parsed["nested"]["safe"] == "visible"


def test_sanitize_payload_handles_lists_and_long_values():
    payload = {
        "items": [{"api_key": "secret"}, {"message": "x" * 300}],
    }

    sanitized = sanitize_payload(payload)

    assert sanitized["items"][0]["api_key"] == "***redacted***"
    assert str(sanitized["items"][1]["message"]).endswith("...")
    assert len(str(sanitized["items"][1]["message"])) < 220
