from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


HOST = "127.0.0.1"
STRUCTURED_PORT = 9011
REPLY_PORT = 9012
CHAT_PATH = "/chat/completions"


@dataclass(frozen=True)
class MiniMindService:
    key: str
    label: str
    model: str
    lora_weight: str
    port: int
    project_root: Path

    @property
    def minimind_root(self) -> Path:
        return self.project_root.parent / "minimind-ticket"

    @property
    def scripts_dir(self) -> Path:
        return self.minimind_root / "scripts"

    @property
    def base_url(self) -> str:
        return f"http://{HOST}:{self.port}/v1"

    def with_port(self, port: int) -> "MiniMindService":
        return MiniMindService(
            key=self.key,
            label=self.label,
            model=self.model,
            lora_weight=self.lora_weight,
            port=port,
            project_root=self.project_root,
        )

    def command(self, python_executable: str | None = None) -> list[str]:
        return [
            python_executable or sys.executable,
            "serve_openai_api.py",
            "--load_from",
            "..\\model",
            "--weight",
            "full_sft",
            "--lora_weight",
            self.lora_weight,
            "--device",
            "cuda",
            "--host",
            HOST,
            "--port",
            str(self.port),
        ]


def build_minimind_services(project_root: Path) -> list[MiniMindService]:
    return [
        MiniMindService(
            key="structured",
            label="结构化模型服务",
            model="minimind-structured-v3",
            lora_weight="lora_ticket_structured_v3",
            port=STRUCTURED_PORT,
            project_root=project_root,
        ),
        MiniMindService(
            key="reply",
            label="回复生成服务",
            model="minimind-reply-v1",
            lora_weight="lora_ticket_reply_v1",
            port=REPLY_PORT,
            project_root=project_root,
        ),
    ]


def runtime_dir(project_root: Path) -> Path:
    return project_root / "data" / "generated" / "runtime"


def log_dir(project_root: Path) -> Path:
    return project_root.parent / "minimind-ticket" / "run_logs"


def registry_path(project_root: Path) -> Path:
    return runtime_dir(project_root) / "minimind_processes.json"


def is_port_open(port: int, host: str = HOST, timeout: float = 0.25) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _load_registry(project_root: Path) -> dict[str, Any]:
    path = registry_path(project_root)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _save_registry(project_root: Path, payload: dict[str, Any]) -> None:
    path = registry_path(project_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _service_log_paths(project_root: Path, service: MiniMindService) -> tuple[Path, Path]:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs = log_dir(project_root)
    logs.mkdir(parents=True, exist_ok=True)
    return (
        logs / f"{service.key}_{service.port}_{stamp}.out.log",
        logs / f"{service.key}_{service.port}_{stamp}.err.log",
    )


def probe_service(service: MiniMindService, timeout: float = 4.0) -> dict[str, Any]:
    url = f"{service.base_url}{CHAT_PATH}"
    try:
        response = requests.post(
            url,
            headers={"Authorization": "Bearer sk-local", "Content-Type": "application/json"},
            json={
                "model": service.model,
                "stream": False,
                "temperature": 0.0,
                "max_tokens": 8,
                "messages": [{"role": "user", "content": "OK"}],
            },
            timeout=timeout,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - health UI should report any connectivity issue.
        return {"ok": False, "message": str(exc)}
    return {"ok": True, "message": "接口可用"}


def service_status(project_root: Path, service: MiniMindService, *, probe: bool = False) -> dict[str, Any]:
    registry = _load_registry(project_root)
    managed = registry.get(service.key, {})
    running = is_port_open(service.port)
    payload: dict[str, Any] = {
        "key": service.key,
        "label": service.label,
        "model": service.model,
        "base_url": service.base_url,
        "port": service.port,
        "running": running,
        "managed": bool(managed and managed.get("pid")),
        "pid": managed.get("pid"),
        "stdout_log": managed.get("stdout_log"),
        "stderr_log": managed.get("stderr_log"),
    }
    if probe and running:
        payload["probe"] = probe_service(service)
    return payload


def runtime_report(project_root: Path, *, probe: bool = False) -> list[dict[str, Any]]:
    return [service_status(project_root, service, probe=probe) for service in build_minimind_services(project_root)]


def start_service(project_root: Path, service: MiniMindService, *, python_executable: str | None = None) -> dict[str, Any]:
    if is_port_open(service.port):
        return {
            "key": service.key,
            "status": "already_running",
            "managed": False,
            "message": f"{service.label} 已在 {service.port} 端口运行。",
        }

    stdout_path, stderr_path = _service_log_paths(project_root, service)
    stdout_handle = stdout_path.open("a", encoding="utf-8")
    stderr_handle = stderr_path.open("a", encoding="utf-8")
    creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) if os.name == "nt" else 0
    try:
        process = subprocess.Popen(
            service.command(python_executable),
            cwd=str(service.scripts_dir),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            shell=False,
            creationflags=creationflags,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()

    registry = _load_registry(project_root)
    registry[service.key] = {
        "pid": process.pid,
        "port": service.port,
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }
    _save_registry(project_root, registry)
    return {
        "key": service.key,
        "status": "started",
        "managed": True,
        "pid": process.pid,
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }


def start_all(project_root: Path, *, python_executable: str | None = None) -> list[dict[str, Any]]:
    return [start_service(project_root, service, python_executable=python_executable) for service in build_minimind_services(project_root)]


def _pid_listening_on_port(port: int) -> int | None:
    if os.name == "nt":
        command = (
            f"Get-NetTCPConnection -LocalAddress {HOST} -LocalPort {int(port)} "
            "-State Listen -ErrorAction SilentlyContinue | "
            "Select-Object -First 1 -ExpandProperty OwningProcess"
        )
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
        )
    else:
        result = subprocess.run(
            ["lsof", "-nP", f"-iTCP:{int(port)}", "-sTCP:LISTEN", "-t"],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
        )
    output = result.stdout.strip().splitlines()
    if not output:
        return None
    try:
        return int(output[0].strip())
    except ValueError:
        return None


def _command_line_for_pid(pid: int) -> str:
    if os.name == "nt":
        command = f"(Get-CimInstance Win32_Process -Filter \"ProcessId={int(pid)}\").CommandLine"
        result = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            check=False,
            capture_output=True,
            text=True,
            timeout=6,
        )
        return result.stdout.strip()
    cmdline_path = Path("/proc") / str(pid) / "cmdline"
    if cmdline_path.exists():
        return cmdline_path.read_text(encoding="utf-8", errors="ignore").replace("\x00", " ").strip()
    return ""


def _is_expected_minimind_process(command_line: str, service: MiniMindService) -> bool:
    normalized = " ".join(command_line.lower().split())
    expected_port = str(service.port)
    has_expected_port = f"--port {expected_port}" in normalized or f"--port={expected_port}" in normalized
    return (
        "serve_openai_api.py" in normalized
        and "--lora_weight" in normalized
        and service.lora_weight.lower() in normalized
        and has_expected_port
    )


def _taskkill(pid: int) -> None:
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], check=False, capture_output=True, text=True, timeout=8)
    else:
        subprocess.run(["kill", str(pid)], check=False, capture_output=True, text=True, timeout=8)


def stop_service(project_root: Path, service: MiniMindService) -> dict[str, Any]:
    registry = _load_registry(project_root)
    managed = registry.get(service.key)
    if not managed or not managed.get("pid"):
        if is_port_open(service.port):
            external_pid = _pid_listening_on_port(service.port)
            command_line = _command_line_for_pid(external_pid) if external_pid else ""
            if external_pid is not None and _is_expected_minimind_process(command_line, service):
                _taskkill(external_pid)
                return {
                    "key": service.key,
                    "status": "stopped_external_safe",
                    "managed": False,
                    "pid": external_pid,
                    "message": f"已停止 {service.label} 的安全匹配外部进程。",
                }
        return {
            "key": service.key,
            "status": "external_or_not_running" if is_port_open(service.port) else "not_running",
            "managed": False,
            "message": "没有找到由本控制台启动的进程，也没有发现可安全匹配的 MiniMind 服务。",
        }

    pid = int(managed["pid"])
    _taskkill(pid)
    registry.pop(service.key, None)
    _save_registry(project_root, registry)
    return {"key": service.key, "status": "stopped", "managed": True, "pid": pid}


def stop_all(project_root: Path) -> list[dict[str, Any]]:
    return [stop_service(project_root, service) for service in build_minimind_services(project_root)]
