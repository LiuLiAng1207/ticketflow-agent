from __future__ import annotations

import asyncio
import datetime as dt
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client


@dataclass(slots=True)
class MCPEmailClient:
    project_root: Path
    env_overrides: dict[str, str]

    async def _call_tool_async(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        env = os.environ.copy()
        env.update(self.env_overrides)
        params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ticketflow.email_mcp_server"],
            cwd=str(self.project_root),
            env=env,
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(tool_name, payload, read_timeout_seconds=dt.timedelta(seconds=45))
        if result.isError:
            raise RuntimeError(f"MCP tool {tool_name} 执行失败：{result.content}")
        if result.structuredContent:
            return dict(result.structuredContent)
        if result.content and hasattr(result.content[0], "text"):
            return {"status": "sent", "content": result.content[0].text}
        return {"status": "sent"}

    def call_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._call_tool_async(tool_name, payload))
