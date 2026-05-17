from __future__ import annotations

import json
from typing import Any

import pytest

from ticketflow.llm import OpenAICompatClient


class _Response:
    status_code = 200
    headers: dict[str, str] = {}

    def __init__(self, payload: dict[str, Any]):
        self.text = json.dumps(payload)

    def raise_for_status(self) -> None:
        return None


def test_deepseek_json_calls_disable_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*_: Any, **kwargs: Any) -> _Response:
        captured["json"] = kwargs["json"]
        return _Response({"choices": [{"message": {"content": '{"status":"ok"}'}}]})

    monkeypatch.setattr("ticketflow.llm.requests.post", fake_post)

    client = OpenAICompatClient(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-v4-flash",
        source_label="llm_cloud",
    )

    assert client.chat_json("system", "Return JSON") == {"status": "ok"}
    assert captured["json"]["response_format"] == {"type": "json_object"}
    assert captured["json"]["thinking"] == {"type": "disabled"}


def test_text_calls_keep_deepseek_thinking_default(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    def fake_post(*_: Any, **kwargs: Any) -> _Response:
        captured["json"] = kwargs["json"]
        return _Response({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("ticketflow.llm.requests.post", fake_post)

    client = OpenAICompatClient(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-v4-flash",
        source_label="llm_cloud",
    )

    assert client.chat_text("system", "hello") == "ok"
    assert "thinking" not in captured["json"]


def test_json_mode_rejects_reasoning_without_content() -> None:
    client = OpenAICompatClient(
        base_url="https://api.deepseek.com",
        api_key="test-key",
        model="deepseek-v4-flash",
        source_label="llm_cloud",
    )
    raw = json.dumps({"choices": [{"message": {"content": "", "reasoning_content": "not json"}}]})

    with pytest.raises(ValueError, match="content"):
        client._extract_message_content(raw, allow_reasoning_content=False)


def test_chat_json_accepts_markdown_wrapped_json(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(*_: Any, **kwargs: Any) -> _Response:
        return _Response({"choices": [{"message": {"content": '```json\n{"status":"ok"}\n```'}}]})

    monkeypatch.setattr("ticketflow.llm.requests.post", fake_post)

    client = OpenAICompatClient(
        base_url="https://api.example.com",
        api_key="test-key",
        model="example-model",
        source_label="llm_cloud",
    )

    assert client.chat_json("system", "Return JSON") == {"status": "ok"}
