from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ticketflow.graph import TicketFlowRunner
from ticketflow.worker import deliver_outbox_event_now, run_ticket_workflow_now


def test_worker_records_failed_workflow_observability_event(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    runner.reset_demo_data()
    task = runner.repository.create_workflow_task(ticket_id="UNKNOWN-TICKET", mode="async")
    runner.close()

    result = run_ticket_workflow_now(
        task_id=task["task_id"],
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "OBSERVABILITY_ENABLED": "true"},
    )

    reloaded = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    events = reloaded.repository.list_observability_events(limit=20)
    reloaded.close()

    assert result["status"] == "failed"
    assert any(
        event["component"] == "worker"
        and event["span_name"] == "run_ticket_workflow"
        and event["status"] == "error"
        and event["error_type"] == "worker_error"
        for event in events
    )


def test_outbox_delivery_failure_records_observability_event(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(
        ticketflow_project,
        overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )
    runner.reset_demo_data()
    ticket_id = runner.list_open_tickets(limit=1)[0].ticket_id
    event = runner.repository.create_outbox_event(
        ticket_id=ticket_id,
        operation_type="incident_email",
        business_key=f"incident-error:{ticket_id}",
        payload={"simulate_error": "SMTP timeout"},
    )
    runner.close()

    deliver_outbox_event_now(
        event_id=event["event_id"],
        project_root=ticketflow_project,
        runner_overrides={"RAG_EMBED_BACKEND": "hash", "ENABLE_PRODUCTION_SERVICES": "true"},
    )

    reloaded = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    events = reloaded.repository.list_observability_events(limit=20)
    reloaded.close()

    assert any(
        item["component"] == "outbox"
        and item["status"] == "error"
        and item["error_type"] == "outbox_error"
        for item in events
    )


def _mcp_env(ticketflow_project: Path, *, write_tools: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path.cwd() / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["TICKETFLOW_PROJECT_ROOT"] = str(ticketflow_project)
    env["RAG_EMBED_BACKEND"] = "hash"
    env["KG_BACKEND"] = "memory"
    env["OBSERVABILITY_ENABLED"] = "true"
    env["TICKETFLOW_MCP_ALLOWED_SCOPES"] = "read,eval"
    env["TICKETFLOW_MCP_ENABLE_WRITE_TOOLS"] = "true" if write_tools else "false"
    env["TICKETFLOW_MCP_ACTOR"] = "pytest-mcp"
    return env


async def _with_mcp_session(ticketflow_project: Path, operation, *, write_tools: bool = False):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "ticketflow.platform_mcp_server"],
        cwd=str(Path.cwd()),
        env=_mcp_env(ticketflow_project, write_tools=write_tools),
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await operation(session)


def test_mcp_tool_call_records_success_and_permission_denied_events(ticketflow_project):
    async def scenario(session: ClientSession):
        await session.call_tool(
            "ticketflow.tickets.list",
            {"limit": 1},
            read_timeout_seconds=dt.timedelta(seconds=30),
        )
        await session.call_tool(
            "ticketflow.workflow.run",
            {"ticket_id": "TCK-0001"},
            read_timeout_seconds=dt.timedelta(seconds=30),
        )

    asyncio.run(_with_mcp_session(ticketflow_project, scenario))

    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    events = runner.repository.list_observability_events(limit=50)
    runner.close()

    assert any(
        event["component"] == "mcp"
        and event["span_name"] == "ticketflow.tickets.list"
        and event["status"] == "ok"
        for event in events
    )
    assert any(
        event["component"] == "mcp"
        and event["span_name"] == "ticketflow.workflow.run"
        and event["status"] == "error"
        and event["error_type"] == "mcp_error"
        for event in events
    )
