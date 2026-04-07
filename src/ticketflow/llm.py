from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(slots=True)
class OpenAICompatClient:
    base_url: str
    api_key: str
    model: str
    source_label: str
    chat_path: str = "/chat/completions"
    verify: str | bool = True
    timeout: int = 30
    max_tokens: int = 256
    temperature: float = 0.0
    max_retries: int = 2
    backoff_seconds: float = 1.5

    def chat_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        content = self._chat_completion(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            response_format={"type": "json_object"},
        )
        return json.loads(content)

    def chat_text(self, system_prompt: str, user_prompt: str) -> str:
        return self._chat_completion(system_prompt=system_prompt, user_prompt=user_prompt)

    def _chat_completion(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        url = f"{self.base_url.rstrip('/')}/{self.chat_path.strip('/')}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        request_json = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": False,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if response_format is not None:
            request_json["response_format"] = response_format

        for attempt in range(self.max_retries + 1):
            response = requests.post(
                url,
                headers=headers,
                json=request_json,
                timeout=self.timeout,
                verify=self.verify,
            )
            if response.status_code not in {429, 500, 502, 503, 504}:
                response.raise_for_status()
                return self._extract_message_content(response.text)

            if attempt >= self.max_retries:
                response.raise_for_status()

            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    sleep_seconds = max(float(retry_after), self.backoff_seconds)
                except ValueError:
                    sleep_seconds = self.backoff_seconds * (2**attempt)
            else:
                sleep_seconds = self.backoff_seconds * (2**attempt)
            time.sleep(sleep_seconds)

        raise RuntimeError("Unreachable retry loop exit in OpenAICompatClient._chat_completion")

    def _extract_message_content(self, raw_text: str) -> str:
        raw_text = raw_text.strip()
        if not raw_text:
            raise ValueError("LLM 返回了空响应。")

        if raw_text.startswith("data:"):
            chunks: list[str] = []
            for line in raw_text.splitlines():
                stripped = line.strip()
                if not stripped.startswith("data:"):
                    continue
                payload_text = stripped[5:].strip()
                if not payload_text or payload_text == "[DONE]":
                    continue
                payload = json.loads(payload_text)
                delta = payload.get("choices", [{}])[0].get("delta", {})
                if delta.get("content"):
                    chunks.append(str(delta["content"]))
                elif delta.get("reasoning_content"):
                    chunks.append(str(delta["reasoning_content"]))
            content = "".join(chunks).strip()
            if not content:
                raise ValueError("LLM 仅返回了空的流式正文。")
            return content

        payload = json.loads(raw_text)
        message = payload.get("choices", [{}])[0].get("message", {})
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()

        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str) and reasoning.strip():
            return reasoning.strip()

        raise ValueError("LLM 返回中没有可用的 content。")
