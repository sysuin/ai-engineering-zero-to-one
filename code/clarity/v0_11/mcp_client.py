"""
Clarity v0.11 — consuming an MCP server from ordinary synchronous code.

The protocol is async and Clarity's agent is not, which is the practical problem
everybody hits second. The answer is not to rewrite the agent: it is to give the
session a thread of its own and talk to it across that boundary.

What comes back is a list of the same `Tool` objects Chapter 16 defined, so nothing
upstream can tell whether a tool is a local function or a process on another machine.
That is the point of a protocol, and it is also the danger — §19.12 measures what
happens when the thing on the other end is not friendly.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.v0_8.tools import Tool, ToolError            # noqa: E402


class Connection:
    """
    One MCP server, held open, callable from synchronous code.

    Opening a session per call would be correct and unusably slow: every call would
    pay for a process start, an `initialize` round trip and an index load. So the
    session lives on a background loop for as long as this object does.
    """

    def __init__(self, command: str, args: list[str], errlog_path: str) -> None:
        self._params = StdioServerParameters(command=command, args=args)
        self._errlog_path = errlog_path
        self._loop = asyncio.new_event_loop()
        self._ready = threading.Event()
        self._stop = asyncio.Event()
        self._session: ClientSession | None = None
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=60)

    def _serve(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._loop.run_until_complete(self._hold())

    async def _hold(self) -> None:
        with open(self._errlog_path, "w") as errlog:
            async with stdio_client(self._params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    self._session = session
                    self._ready.set()
                    await self._stop.wait()

    def _run(self, coroutine) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(60)

    def tools(self) -> list[Tool]:
        """Ask the server what it can do, and wrap each answer as a Clarity tool."""
        listed = self._run(self._session.list_tools())
        return [self._wrap(t) for t in listed.tools]

    def _wrap(self, spec) -> Tool:
        def call(**arguments):
            result = self._run(self._session.call_tool(spec.name, arguments))
            text = "\n".join(block.text for block in result.content
                             if hasattr(block, "text"))
            if result.is_error:
                # The server's error becomes the model's error, unchanged. Rewriting
                # it here would throw away the one thing §19.8 says is useful.
                raise ToolError(text)
            if result.structured_content is not None:
                return result.structured_content.get("result",
                                                     result.structured_content)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text

        return Tool(spec.name, spec.description or "", spec.input_schema, call)

    def close(self) -> None:
        self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=30)
