from __future__ import annotations

from pathlib import Path

from ticketflow.db import TicketFlowRepository
from ticketflow.migrations import run_migrations
from ticketflow.repository import RepositoryProtocol, create_repository
from ticketflow.service_settings import ServiceSettings


def _repo(tmp_path: Path) -> TicketFlowRepository:
    repository = TicketFlowRepository(tmp_path / "ticketflow.sqlite3")
    repository.init_database()
    return repository


def test_repository_tracks_workflow_task_lifecycle(tmp_path):
    repository = _repo(tmp_path)

    task = repository.create_workflow_task(ticket_id="TCK-001", mode="async", thread_id="thread-tck-001")
    assert task["status"] == "queued"
    assert task["ticket_id"] == "TCK-001"

    running = repository.mark_workflow_task_running(task["task_id"], celery_task_id="celery-1")
    assert running["status"] == "running"
    assert running["celery_task_id"] == "celery-1"

    completed = repository.complete_workflow_task(task["task_id"], result={"ok": True})
    assert completed["status"] == "succeeded"
    assert completed["result"]["ok"] is True

    loaded = repository.get_workflow_task(task["task_id"])
    assert loaded is not None
    assert loaded["status"] == "succeeded"


def test_repository_records_failed_and_cancelled_workflow_tasks(tmp_path):
    repository = _repo(tmp_path)
    failed_task = repository.create_workflow_task(ticket_id="TCK-002", mode="async")
    cancelled_task = repository.create_workflow_task(ticket_id="TCK-003", mode="async")

    failed = repository.fail_workflow_task(failed_task["task_id"], error_message="boom")
    cancelled = repository.cancel_workflow_task(cancelled_task["task_id"], reason="operator_cancelled")

    assert failed["status"] == "failed"
    assert failed["error_message"] == "boom"
    assert cancelled["status"] == "cancelled"
    assert cancelled["error_message"] == "operator_cancelled"


def test_outbox_event_is_idempotent_by_business_key(tmp_path):
    repository = _repo(tmp_path)

    first = repository.create_outbox_event(
        ticket_id="TCK-004",
        operation_type="incident_email",
        business_key="incident:TCK-004",
        payload={"recipient": "ops@example.com"},
    )
    second = repository.create_outbox_event(
        ticket_id="TCK-004",
        operation_type="incident_email",
        business_key="incident:TCK-004",
        payload={"recipient": "ops@example.com"},
    )

    assert first["event_id"] == second["event_id"]
    assert first["deduplicated"] is False
    assert second["deduplicated"] is True
    assert len(repository.list_outbox_events()) == 1


def test_idempotency_keys_and_external_locks_are_persistent(tmp_path):
    repository = _repo(tmp_path)

    first_key = repository.record_idempotency_key(
        key_scope="refund",
        key_value="TCK-005:ORDER-001",
        request_hash="hash-1",
        response_payload={"refund_id": 1},
    )
    second_key = repository.record_idempotency_key(
        key_scope="refund",
        key_value="TCK-005:ORDER-001",
        request_hash="hash-1",
        response_payload={"refund_id": 1},
    )
    first_lock = repository.acquire_external_operation_lock(
        lock_key="refund:TCK-005:ORDER-001",
        ticket_id="TCK-005",
        operation_type="refund",
        acquired_by="worker-1",
    )
    second_lock = repository.acquire_external_operation_lock(
        lock_key="refund:TCK-005:ORDER-001",
        ticket_id="TCK-005",
        operation_type="refund",
        acquired_by="worker-2",
    )

    assert first_key["deduplicated"] is False
    assert second_key["deduplicated"] is True
    assert first_lock["acquired"] is True
    assert second_lock["acquired"] is False


def test_approval_request_and_decision_are_durable(tmp_path):
    repository = _repo(tmp_path)

    approval = repository.create_approval_request(
        ticket_id="TCK-006",
        thread_id="thread-tck-006",
        tool_name="issue_refund_request",
        tool_args={"order_id": "ORDER-001"},
        payload={"route_family": "refund_candidate"},
        requested_by="workflow",
    )
    decision = repository.record_approval_decision(
        approval_id=approval["approval_id"],
        decision="approve",
        reviewer="lead",
        comment="证据充分，同意提交退款审批。",
    )

    loaded = repository.get_approval_request(approval["approval_id"])
    assert loaded is not None
    assert loaded["status"] == "approved"
    assert decision["approval_id"] == approval["approval_id"]
    assert repository.list_approval_requests(status="approved")[0]["approval_id"] == approval["approval_id"]


def test_sqlite_repository_satisfies_repository_protocol(tmp_path):
    repository = _repo(tmp_path)

    assert isinstance(repository, RepositoryProtocol)


def test_repository_factory_defaults_to_sqlite(tmp_path):
    settings = ServiceSettings(
        project_root=tmp_path,
        app_env="test",
        api_host="127.0.0.1",
        api_port=8000,
        database_backend="sqlite",
        database_url=None,
        redis_url=None,
        qdrant_url=None,
        neo4j_uri=None,
        minio_endpoint=None,
        enable_production_services=False,
        celery_broker_url=None,
        celery_result_backend=None,
        celery_task_always_eager=True,
    )

    repository = create_repository(settings)

    assert isinstance(repository, TicketFlowRepository)


def test_migration_runner_initializes_sqlite_control_tables(tmp_path):
    result = run_migrations(project_root=tmp_path, backend="sqlite")
    repository = TicketFlowRepository(Path(result["sqlite_path"]))

    task = repository.create_workflow_task(ticket_id="TCK-MIGRATE", mode="async")

    assert result["backend"] == "sqlite"
    assert task["status"] == "queued"
