from __future__ import annotations

from fastapi.testclient import TestClient

from ticketflow.api import create_app


def _client(ticketflow_project):
    app = create_app(
        project_root=ticketflow_project,
        runner_overrides={
            "RAG_EMBED_BACKEND": "hash",
            "KG_BACKEND": "memory",
            "CELERY_TASK_ALWAYS_EAGER": "true",
        },
    )
    with TestClient(app) as client:
        yield client


def test_claw_api_reload_list_run_and_leaderboard(ticketflow_project):
    client = next(_client(ticketflow_project))

    reload_response = client.post("/api/v1/claw/tasks/reload")
    listing = client.get("/api/v1/claw/tasks")
    detail = client.get("/api/v1/claw/tasks/claw-query-ticket-evidence")
    sync_run = client.post("/api/v1/claw/tasks/claw-query-ticket-evidence/run", params={"sync": "true", "pass_k": 2})
    run_id = sync_run.json()["run"]["run_id"]
    run_detail = client.get(f"/api/v1/claw/runs/{run_id}")
    leaderboard = client.get("/api/v1/claw/leaderboard")

    assert reload_response.status_code == 200
    assert reload_response.json()["loaded_count"] >= 5
    assert listing.status_code == 200
    assert any(task["task_id"] == "claw-query-ticket-evidence" for task in listing.json()["tasks"])
    assert detail.status_code == 200
    assert detail.json()["task"]["task_id"] == "claw-query-ticket-evidence"
    assert sync_run.status_code == 200
    assert sync_run.json()["run"]["status"] == "succeeded"
    assert sync_run.json()["run"]["summary"]["attempt_count"] == 2
    assert run_detail.status_code == 200
    assert len(run_detail.json()["run"]["attempts"]) == 2
    assert leaderboard.status_code == 200
    assert any(row["task_id"] == "claw-query-ticket-evidence" for row in leaderboard.json()["leaderboard"])


def test_claw_api_async_run_and_unknown_task(ticketflow_project):
    client = next(_client(ticketflow_project))
    client.post("/api/v1/claw/tasks/reload")

    async_run = client.post("/api/v1/claw/tasks/claw-query-ticket-evidence/run", params={"pass_k": 1})
    missing_detail = client.get("/api/v1/claw/tasks/missing-task")
    missing_run = client.post("/api/v1/claw/tasks/missing-task/run")

    assert async_run.status_code == 202
    assert async_run.json()["task_id"] == "claw-query-ticket-evidence"
    assert async_run.json()["run_id"]
    assert missing_detail.status_code == 404
    assert missing_run.status_code == 404
