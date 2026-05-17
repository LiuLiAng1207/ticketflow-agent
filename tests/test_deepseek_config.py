from __future__ import annotations

from pathlib import Path


def test_deepseek_aliases_enable_cloud_backend(tmp_path: Path, monkeypatch) -> None:
    from ticketflow.config import TicketFlowSettings

    for name in (
        "MODEL_BACKEND",
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_MODEL",
        "OPENAI_COMPAT_CHAT_PATH",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "MODEL_BACKEND=cloud_api",
                "DEEPSEEK_API_KEY=sk-test",
            ]
        ),
        encoding="utf-8",
    )

    settings = TicketFlowSettings.from_project_root(tmp_path)

    assert settings.model_backend == "cloud_api"
    assert settings.cloud_api_key == "sk-test"
    assert settings.cloud_api_base_url == "https://api.deepseek.com"
    assert settings.cloud_model_name == "deepseek-v4-flash"
    assert settings.cloud_api_chat_path == "/chat/completions"
    assert settings.cloud_llm_enabled


def test_deepseek_model_alias_can_override_default(tmp_path: Path, monkeypatch) -> None:
    from ticketflow.config import TicketFlowSettings

    for name in (
        "OPENAI_COMPAT_API_KEY",
        "OPENAI_COMPAT_BASE_URL",
        "OPENAI_COMPAT_MODEL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "MODEL_BACKEND=cloud_api",
                "DEEPSEEK_API_KEY=sk-test",
                "DEEPSEEK_MODEL=deepseek-reasoner",
                "DEEPSEEK_BASE_URL=https://api.deepseek.com/v1",
            ]
        ),
        encoding="utf-8",
    )

    settings = TicketFlowSettings.from_project_root(tmp_path)

    assert settings.cloud_model_name == "deepseek-reasoner"
    assert settings.cloud_api_base_url == "https://api.deepseek.com/v1"
