from __future__ import annotations

import socket
import subprocess
from pathlib import Path

import pytest


def _reserve_port() -> socket.socket:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind(("127.0.0.1", 0))
    sock.listen(1)
    return sock


def test_minimind_service_specs_are_fixed_to_interview_ports(tmp_path: Path):
    from ticketflow.runtime_control import build_minimind_services

    project_root = tmp_path / "ticketflow"
    project_root.mkdir()
    services = build_minimind_services(project_root)

    assert [service.key for service in services] == ["structured", "reply"]
    assert [service.port for service in services] == [9011, 9012]
    assert services[0].lora_weight == "lora_ticket_structured_v3"
    assert services[1].lora_weight == "lora_ticket_reply_v1"
    assert services[0].command("python")[0] == "python"
    assert "shell=True" not in " ".join(services[0].command("python"))


def test_start_service_does_not_spawn_when_port_is_already_busy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ticketflow.runtime_control import build_minimind_services, start_service

    project_root = tmp_path / "ticketflow"
    project_root.mkdir()
    service = build_minimind_services(project_root)[0]
    sock = _reserve_port()
    service = service.with_port(sock.getsockname()[1])

    def fail_popen(*_args, **_kwargs):
        raise AssertionError("Popen should not be called when the service port is already running")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    try:
        result = start_service(project_root, service, python_executable="python")
    finally:
        sock.close()

    assert result["status"] == "already_running"
    assert result["managed"] is False


def test_start_service_uses_fixed_command_without_shell(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from ticketflow.runtime_control import build_minimind_services, start_service

    project_root = tmp_path / "ticketflow"
    project_root.mkdir()
    service = build_minimind_services(project_root)[0].with_port(65500)
    captured: dict[str, object] = {}

    class FakeProcess:
        pid = 43210

    def fake_popen(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return FakeProcess()

    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    result = start_service(project_root, service, python_executable="python")

    assert result["status"] == "started"
    assert result["pid"] == 43210
    assert captured["kwargs"]["shell"] is False
    assert captured["kwargs"]["cwd"] == str(service.scripts_dir)
    assert captured["args"] == service.command("python")


def test_stop_service_can_stop_safe_external_minimind_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ticketflow.runtime_control as runtime_control

    project_root = tmp_path / "ticketflow"
    project_root.mkdir()
    service = runtime_control.build_minimind_services(project_root)[0]
    killed: list[int] = []

    monkeypatch.setattr(runtime_control, "is_port_open", lambda port: port == service.port)
    monkeypatch.setattr(runtime_control, "_pid_listening_on_port", lambda port: 12345 if port == service.port else None)
    monkeypatch.setattr(
        runtime_control,
        "_command_line_for_pid",
        lambda pid: (
            r"D:\Anaconda\python.exe serve_openai_api.py --load_from ..\model "
            r"--weight full_sft --lora_weight lora_ticket_structured_v3 "
            r"--device cuda --host 127.0.0.1 --port 9011"
        )
        if pid == 12345
        else "",
    )
    monkeypatch.setattr(runtime_control, "_taskkill", lambda pid: killed.append(pid))

    result = runtime_control.stop_service(project_root, service)

    assert result["status"] == "stopped_external_safe"
    assert result["managed"] is False
    assert result["pid"] == 12345
    assert killed == [12345]


def test_stop_service_refuses_unknown_external_python_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import ticketflow.runtime_control as runtime_control

    project_root = tmp_path / "ticketflow"
    project_root.mkdir()
    service = runtime_control.build_minimind_services(project_root)[0]
    killed: list[int] = []

    monkeypatch.setattr(runtime_control, "is_port_open", lambda port: port == service.port)
    monkeypatch.setattr(runtime_control, "_pid_listening_on_port", lambda port: 12345 if port == service.port else None)
    monkeypatch.setattr(runtime_control, "_command_line_for_pid", lambda pid: r"D:\Anaconda\python.exe other_server.py --port 9011")
    monkeypatch.setattr(runtime_control, "_taskkill", lambda pid: killed.append(pid))

    result = runtime_control.stop_service(project_root, service)

    assert result["status"] == "external_or_not_running"
    assert result["managed"] is False
    assert killed == []


def test_taskkill_uses_timeout_on_windows(monkeypatch: pytest.MonkeyPatch):
    import ticketflow.runtime_control as runtime_control

    captured: dict[str, object] = {}

    def fake_run(args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr(runtime_control.os, "name", "nt")
    monkeypatch.setattr(runtime_control.subprocess, "run", fake_run)

    runtime_control._taskkill(12345)

    assert captured["args"] == ["taskkill", "/PID", "12345", "/T", "/F"]
    assert captured["kwargs"]["timeout"] == 8
