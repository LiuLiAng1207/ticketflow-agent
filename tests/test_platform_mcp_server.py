from __future__ import annotations

import asyncio
import datetime as dt
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from ticketflow.graph import TicketFlowRunner


def _mcp_env(ticketflow_project: Path, *, write_tools: bool = False) -> dict[str, str]:
    env = os.environ.copy()
    src_path = str(Path.cwd() / "src")
    env["PYTHONPATH"] = src_path + os.pathsep + env.get("PYTHONPATH", "")
    env["TICKETFLOW_PROJECT_ROOT"] = str(ticketflow_project)
    env["RAG_EMBED_BACKEND"] = "hash"
    env["KG_BACKEND"] = "memory"
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


def test_platform_mcp_lists_core_tools_and_blocks_write_tools_by_default(ticketflow_project):
    async def scenario(session: ClientSession):
        tools = await session.list_tools()
        names = {tool.name for tool in tools.tools}
        denied = await session.call_tool(
            "ticketflow.workflow.run",
            {"ticket_id": "TCK-0001"},
            read_timeout_seconds=dt.timedelta(seconds=30),
        )
        return names, denied

    tool_names, denied_result = asyncio.run(_with_mcp_session(ticketflow_project, scenario))

    assert {
        "ticketflow.tickets.list",
        "ticketflow.tickets.get",
        "ticketflow.kg.search",
        "ticketflow.claw.leaderboard",
        "ticketflow.agent.chat",
    }.issubset(tool_names)
    assert denied_result.isError is True
    assert "写入类 MCP 工具未启用" in denied_result.content[0].text


def test_platform_mcp_read_tools_resources_and_prompts(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    runner.reset_demo_data()
    ticket_id = runner.list_open_tickets(limit=1)[0].ticket_id
    runner.close()

    async def scenario(session: ClientSession):
        listing = await session.call_tool(
            "ticketflow.tickets.list",
            {"limit": 3},
            read_timeout_seconds=dt.timedelta(seconds=30),
        )
        ticket = await session.call_tool(
            "ticketflow.tickets.get",
            {"ticket_id": ticket_id},
            read_timeout_seconds=dt.timedelta(seconds=30),
        )
        kg = await session.call_tool(
            "ticketflow.kg.search",
            {"query": ticket_id, "limit": 5},
            read_timeout_seconds=dt.timedelta(seconds=30),
        )
        leaderboard = await session.call_tool(
            "ticketflow.claw.leaderboard",
            {"limit": 5},
            read_timeout_seconds=dt.timedelta(seconds=30),
        )
        resource = await session.read_resource(f"ticketflow://tickets/{ticket_id}")
        prompts = await session.list_prompts()
        prompt = await session.get_prompt("explain_ticket", {"ticket_id": ticket_id})
        return listing, ticket, kg, leaderboard, resource, prompts, prompt

    listing, ticket, kg, leaderboard, resource, prompts, prompt = asyncio.run(
        _with_mcp_session(ticketflow_project, scenario)
    )

    assert listing.structuredContent["count"] >= 1
    assert ticket.structuredContent["ticket"]["ticket_id"] == ticket_id
    assert kg.structuredContent["status"] in {"ok", "disabled", "unavailable"}
    assert "leaderboard" in leaderboard.structuredContent
    assert json.loads(resource.contents[0].text)["ticket"]["ticket_id"] == ticket_id
    assert "explain_ticket" in {item.name for item in prompts.prompts}
    assert ticket_id in prompt.messages[0].content.text


def test_platform_mcp_write_enabled_can_create_workflow_task(ticketflow_project):
    runner = TicketFlowRunner.from_project_root(ticketflow_project, overrides={"RAG_EMBED_BACKEND": "hash"})
    runner.reset_demo_data()
    ticket_id = runner.list_open_tickets(limit=1)[0].ticket_id
    runner.close()

    async def scenario(session: ClientSession):
        result = await session.call_tool(
            "ticketflow.workflow.run",
            {"ticket_id": ticket_id},
            read_timeout_seconds=dt.timedelta(seconds=30),
        )
        return result

    result = asyncio.run(_with_mcp_session(ticketflow_project, scenario, write_tools=True))

    assert result.isError is False
    assert result.structuredContent["task"]["ticket_id"] == ticket_id
    assert result.structuredContent["task"]["status"] == "queued"

