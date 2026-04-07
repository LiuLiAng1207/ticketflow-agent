from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import TicketFlowSettings
from .llm import OpenAICompatClient


def _build_client(settings: TicketFlowSettings, target: str) -> OpenAICompatClient | None:
    if target == "cloud" and settings.cloud_llm_enabled:
        return OpenAICompatClient(
            base_url=settings.cloud_api_base_url,
            api_key=settings.cloud_api_key,
            model=settings.cloud_model_name,
            source_label="llm_cloud",
            chat_path=settings.cloud_api_chat_path,
            verify=settings.cloud_api_ca_cert or True,
        )
    if target == "minimind" and settings.minimind_enabled:
        return OpenAICompatClient(
            base_url=settings.minimind_base_url,
            api_key=settings.minimind_api_key,
            model=settings.minimind_model,
            source_label="llm_minimind",
            chat_path=settings.minimind_chat_path,
            verify=True,
        )
    if target == "minimind_structured" and settings.minimind_structured_base_url and settings.minimind_structured_model:
        return OpenAICompatClient(
            base_url=settings.minimind_structured_base_url,
            api_key=settings.minimind_structured_api_key,
            model=settings.minimind_structured_model,
            source_label="llm_minimind",
            chat_path=settings.minimind_chat_path,
            verify=True,
        )
    if target == "minimind_reply" and settings.minimind_reply_base_url and settings.minimind_reply_model:
        return OpenAICompatClient(
            base_url=settings.minimind_reply_base_url,
            api_key=settings.minimind_reply_api_key,
            model=settings.minimind_reply_model,
            source_label="llm_minimind",
            chat_path=settings.minimind_chat_path,
            verify=True,
        )
    return None


def _probe_client(client: OpenAICompatClient, label: str) -> dict[str, Any]:
    payload = client.chat_json(
        system_prompt="你是连通性检查助手，只返回 JSON。",
        user_prompt="请返回一个 JSON，对象字段必须包含 status、provider、message。",
    )
    payload["target"] = label
    return payload


def run_check(project_root: Path, target: str = "active") -> dict[str, Any]:
    settings = TicketFlowSettings.from_project_root(project_root)
    report: dict[str, Any] = {
        "project_root": str(project_root),
        "model_backend": settings.model_backend,
        "rag_enabled": settings.rag_enabled,
        "cloud_ready": settings.cloud_llm_enabled,
        "minimind_ready": settings.minimind_enabled,
        "checks": [],
    }

    if target == "active":
        if settings.model_backend == "cloud_api":
            targets = ["cloud"]
        elif settings.model_backend == "minimind_split":
            targets = ["minimind_structured", "minimind_reply"]
            if settings.cloud_llm_enabled:
                targets.append("cloud")
        elif settings.model_backend == "minimind_api":
            targets = ["minimind"]
            if settings.cloud_llm_enabled:
                targets.append("cloud")
        else:
            report["status"] = "ok"
            report["message"] = "当前处于 rule 模式，无需 LLM 连通性检查。"
            return report
    elif target == "all":
        targets = [name for name in ("cloud", "minimind", "minimind_structured", "minimind_reply") if _build_client(settings, name) is not None]
    else:
        targets = [target]

    if not targets:
        report["status"] = "missing_config"
        report["message"] = "没有找到可检查的模型后端配置。"
        return report

    overall_ok = True
    for item in targets:
        client = _build_client(settings, item)
        if client is None:
            report["checks"].append(
                {"target": item, "status": "missing_config", "message": f"{item} 后端配置不完整。"}
            )
            overall_ok = False
            continue
        try:
            result = _probe_client(client, item)
            report["checks"].append({"target": item, "status": "ok", "response": result})
        except Exception as exc:  # noqa: BLE001
            report["checks"].append({"target": item, "status": "failed", "message": str(exc)})
            overall_ok = False

    report["status"] = "ok" if overall_ok else "failed"
    report["message"] = "所有目标后端都已连通。" if overall_ok else "存在后端连通性失败，请检查配置或网络。"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Check TicketFlow LLM connectivity.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="TicketFlow 项目根目录。",
    )
    parser.add_argument(
        "--target",
        choices=["active", "all", "cloud", "minimind", "minimind_structured", "minimind_reply"],
        default="active",
        help="要检查的模型后端。",
    )
    args = parser.parse_args()

    report = run_check(args.project_root, target=args.target)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["status"] != "ok":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
