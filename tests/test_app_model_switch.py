from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json


def test_runtime_env_overrides_for_deepseek_switch(tmp_path: Path) -> None:
    from ticketflow.app import _runtime_env_overrides

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "MODEL_BACKEND=minimind_split",
                "DEEPSEEK_API_KEY=sk-test",
                "MINIMIND_STRUCTURED_BASE_URL=http://127.0.0.1:9011/v1",
                "MINIMIND_STRUCTURED_MODEL=minimind-structured-v3",
            ]
        ),
        encoding="utf-8",
    )

    overrides = _runtime_env_overrides(tmp_path, "cloud_api")

    assert overrides["MODEL_BACKEND"] == "cloud_api"
    assert overrides["DEEPSEEK_API_KEY"] == "sk-test"
    assert overrides["DEEPSEEK_BASE_URL"] == "https://api.deepseek.com"
    assert overrides["DEEPSEEK_MODEL"] == "deepseek-v4-flash"
    assert overrides["OPENAI_COMPAT_CHAT_PATH"] == "/chat/completions"
    assert overrides["MINIMIND_STRUCTURED_BASE_URL"] == "http://127.0.0.1:9011/v1"


def test_runtime_env_overrides_for_minimind_switch_keeps_local_backend(tmp_path: Path) -> None:
    from ticketflow.app import _runtime_env_overrides

    (tmp_path / ".env").write_text(
        "\n".join(
            [
                "MODEL_BACKEND=cloud_api",
                "DEEPSEEK_API_KEY=sk-test",
                "MINIMIND_REPLY_BASE_URL=http://127.0.0.1:9012/v1",
            ]
        ),
        encoding="utf-8",
    )

    overrides = _runtime_env_overrides(tmp_path, "minimind_split")

    assert overrides["MODEL_BACKEND"] == "minimind_split"
    assert overrides["MINIMIND_REPLY_BASE_URL"] == "http://127.0.0.1:9012/v1"


def test_service_api_base_url_uses_explicit_or_normalized_host(monkeypatch, tmp_path: Path) -> None:
    from ticketflow.app import _service_api_base_url

    monkeypatch.delenv("TICKETFLOW_API_URL", raising=False)
    monkeypatch.delenv("API_BASE_URL", raising=False)
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    (tmp_path / ".env").write_text("API_HOST=0.0.0.0\nAPI_PORT=18000\n", encoding="utf-8")

    assert _service_api_base_url(tmp_path) == "http://127.0.0.1:18000"

    (tmp_path / ".env").write_text("TICKETFLOW_API_URL=http://ticketflow-api:8000/\n", encoding="utf-8")

    assert _service_api_base_url(tmp_path) == "http://ticketflow-api:8000"


def test_control_plane_snapshot_prefers_api(monkeypatch, tmp_path: Path) -> None:
    from ticketflow.app import _load_control_plane_snapshot

    def fake_fetch(base_url: str, path: str, *, timeout: float = 1.5):
        del base_url, timeout
        if path == "/readyz":
            return {"database": {"backend": "postgres"}, "dependencies": {"celery": {"task_always_eager": False}}}
        if path.startswith("/api/v1/tasks"):
            return {"tasks": [{"task_id": "task-api", "status": "queued"}]}
        if path.startswith("/api/v1/approvals"):
            return {"approvals": [{"approval_id": "approval-api", "status": "pending"}]}
        if path.startswith("/api/v1/outbox"):
            return {"events": [{"event_id": "outbox-api", "status": "pending"}]}
        if path == "/api/v1/kg/health":
            return {"status": "ok", "backend": "neo4j", "node_count": 8, "edge_count": 9}
        raise AssertionError(path)

    monkeypatch.setattr("ticketflow.app._fetch_api_json", fake_fetch)
    runner = SimpleNamespace(repository=SimpleNamespace())

    snapshot = _load_control_plane_snapshot(tmp_path, runner)

    assert snapshot["source"] == "api"
    assert snapshot["tasks"][0]["task_id"] == "task-api"
    assert snapshot["approvals"][0]["approval_id"] == "approval-api"
    assert snapshot["outbox_events"][0]["event_id"] == "outbox-api"
    assert snapshot["kg_health"]["backend"] == "neo4j"


def test_control_plane_snapshot_falls_back_to_repository(monkeypatch, tmp_path: Path) -> None:
    from ticketflow.app import _load_control_plane_snapshot

    def broken_fetch(base_url: str, path: str, *, timeout: float = 1.5):
        del base_url, path, timeout
        raise OSError("api offline")

    repository = SimpleNamespace(
        list_workflow_tasks=lambda limit=20: [{"task_id": "task-local", "status": "running"}],
        list_approval_requests=lambda limit=20: [{"approval_id": "approval-local", "status": "pending"}],
        list_outbox_events=lambda limit=20: [{"event_id": "outbox-local", "status": "pending"}],
    )
    monkeypatch.setattr("ticketflow.app._fetch_api_json", broken_fetch)

    snapshot = _load_control_plane_snapshot(tmp_path, SimpleNamespace(repository=repository))

    assert snapshot["source"] == "repository"
    assert "api offline" in snapshot["error"]
    assert snapshot["tasks"][0]["task_id"] == "task-local"
    assert snapshot["approvals"][0]["approval_id"] == "approval-local"
    assert snapshot["outbox_events"][0]["event_id"] == "outbox-local"
    assert snapshot["kg_health"]["status"] == "unknown"


def test_sidebar_keeps_expand_control_visible() -> None:
    app_source = Path("src/ticketflow/app.py").read_text(encoding="utf-8")

    assert 'initial_sidebar_state="expanded"' in app_source
    assert '#MainMenu, .stDeployButton, [data-testid="stAppDeployButton"] {' in app_source
    assert '#MainMenu, .stDeployButton, header [data-testid="stToolbar"]' not in app_source
    assert 'header [data-testid="stToolbar"]' in app_source
    assert "visibility: visible;" in app_source
    assert "停止本控制台启动的进程，或安全匹配的 MiniMind 端口进程" in app_source
    assert "只停止由本控制台启动的进程" not in app_source


def test_sufficiency_summary_hides_raw_english_reasoning() -> None:
    from ticketflow.app import _sufficiency_summary

    result = SimpleNamespace(
        route_family="refund_candidate",
        sufficient=False,
        missing_sources=["policy"],
        required_sources=["policy", "order"],
        supporting_doc_ids=["policy:POL-002-02", "order:ORD-0027"],
        reasoning=(
            "Policy evidence is required but not provided; the policy evidence listed is generic. "
            "Order evidence is present but insufficient without policy."
        ),
    )

    summary = _sufficiency_summary(result)

    assert "订单证据已找到" in summary
    assert "缺少可放行退款的明确策略条款" in summary
    assert "Policy evidence" not in summary
    assert "Order evidence" not in summary


def test_triage_summary_uses_structured_chinese_not_raw_reasoning() -> None:
    from ticketflow.app import _triage_summary

    ticket = SimpleNamespace(customer_tier="premium", channel="chat")
    triage = SimpleNamespace(
        category="account_access",
        priority="urgent",
        urgency="sev1",
        sla_risk=False,
        reasoning=(
            "User is premium via chat. priority=urgent and urgency=sev1 because "
            "account access failed repeatedly."
        ),
    )

    summary = _triage_summary(triage, ticket)

    assert "账号访问" in summary
    assert "高级版客户" in summary
    assert "在线聊天" in summary
    assert "优先级为紧急" in summary
    assert "一级紧急处理" in summary
    assert "premium" not in summary
    assert "chat" not in summary
    assert "urgent" not in summary
    assert "sev1" not in summary


def test_action_summary_uses_structured_chinese_not_raw_rationale() -> None:
    from ticketflow.app import _action_summary

    action = SimpleNamespace(
        action_type="troubleshoot",
        target_status="investigating",
        requires_approval=False,
        suggested_tool="update_ticket_status",
        route_family="standard_resolution",
        sufficiency_passed=True,
        missing_sources=[],
        supporting_doc_ids=["kb-account-access-001", "history-031"],
        rationale=(
            "Ticket is urgent (sev1) for premium customer. Sufficiency check passed "
            "(sufficient=true). Evidence supports password reset and MFA configuration."
        ),
    )
    sufficiency = SimpleNamespace(
        sufficient=True,
        missing_sources=[],
        required_sources=[],
        route_family="standard_resolution",
    )

    summary = _action_summary(action, sufficiency)

    assert "系统建议执行“故障排查”" in summary
    assert "目标状态为“排查中”" in summary
    assert "无需人工审批" in summary
    assert "更新工单状态工具" in summary
    assert "Ticket is urgent" not in summary
    assert "sufficient=true" not in summary
    assert "password reset" not in summary
    assert "MFA" not in summary


def test_run_ticket_for_ui_converts_runtime_exception_to_chinese_message() -> None:
    from ticketflow.app import _run_ticket_for_ui

    class BrokenRunner:
        def run_ticket(self, ticket_id: str) -> None:
            del ticket_id
            raise ConnectionError("cloud endpoint unavailable")

    result, error_message = _run_ticket_for_ui(BrokenRunner(), "TCK-0001")

    assert result is None
    assert error_message is not None
    assert "工作流运行失败" in error_message
    assert "模型服务连接异常" in error_message
    assert "ConnectionError" in error_message
    assert "Traceback" not in error_message


def test_reply_internal_summary_hides_raw_english_internal_note() -> None:
    from ticketflow.app import _reply_internal_summary

    reply = SimpleNamespace(
        status="waiting_on_customer",
        internal_note=(
            "Customer inquiry about pricing and delivery. Sufficiency is high, "
            "but action is low-risk."
        ),
        review_passed=True,
    )
    state = {
        "proposed_action": SimpleNamespace(action_type="request_info"),
        "reply_fact_check": SimpleNamespace(passed=True, supported_claim_ratio=1.0),
    }

    summary = _reply_internal_summary(reply, state)

    assert "客户回复状态为“等待客户补充”" in summary
    assert "对应动作是“补充信息”" in summary
    assert "回复事实校验已通过" in summary
    assert "Customer inquiry" not in summary
    assert "Sufficiency is high" not in summary
    assert "low-risk" not in summary


def test_benchmark_report_cards_read_retrieval_and_multimodal_reports(tmp_path: Path) -> None:
    from ticketflow.app import _benchmark_report_cards

    reports = tmp_path / "reports"
    reports.mkdir()
    (reports / "retrieval_strategy_eval_a40_bge_v8.json").write_text(
        json.dumps(
            {
                "sample_size": 600,
                "metrics": {"hit_rate": 0.889, "mrr": 0.612, "error_rate": 0.111},
                "retrieval_stack": {
                    "embedding_model": "BAAI/bge-m3",
                    "fusion_method": "rrf",
                    "reranker_backend": "bge-reranker-v2-m3",
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (reports / "multimodal_ablation_a40_bge_v2.json").write_text(
        json.dumps(
            {
                "sample_count": 200,
                "enabled": {"metrics": {"multimodal_evidence_recall": 1.0}},
                "text_only": {"metrics": {"multimodal_evidence_recall": 0.0}},
                "lift": {"multimodal_evidence_recall": 1.0, "attachment_entity_accuracy": 0.772},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    cards = _benchmark_report_cards(tmp_path)

    assert cards[0]["title"] == "检索增强专项评测"
    assert cards[0]["metrics"]["HitRate"] == "88.9%"
    assert cards[0]["metrics"]["MRR"] == "0.612"
    assert "BAAI/bge-m3" in cards[0]["caption"]
    assert cards[1]["title"] == "多模态附件消融评测"
    assert cards[1]["metrics"]["证据召回增益"] == "+100.0%"
    assert cards[1]["metrics"]["实体抽取增益"] == "+77.2%"
