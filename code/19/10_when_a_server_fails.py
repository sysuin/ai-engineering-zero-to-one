# timeout: 300
# A server is a process you do not control. What a client sees when a tool hangs and when the
# server dies — and the two lines of client code that turn each into an answer.

import asyncio
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(command=sys.executable, args=["code/19/_fragile_server.py"])


async def timed(label: str, coroutine) -> None:
    started = time.perf_counter()
    try:
        result = await coroutine
        text = result.content[0].text if result.content else ""
        outcome = f"error result: {text[:60]}" if result.is_error else f"returned {text!r}"
    except BaseException as error:                                    # noqa: BLE001
        outcome = f"raised {type(error).__name__}: {str(error)[:60]}"
    print(f"  {label:40} {time.perf_counter() - started:5.1f}s  {outcome}")


async def main() -> None:
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            print("a tool that takes 8 seconds")
            await timed("no timeout", session.call_tool("slow", {"seconds": 8}))
            await timed("read_timeout_seconds=2", session.call_tool(
                "slow", {"seconds": 8}, read_timeout_seconds=2))
            await timed("the next call, on the same session", session.call_tool("ping", {}))

            print("\nthe server process dies during a call")
            await timed("the call that killed it", session.call_tool("crash", {}, read_timeout_seconds=5))
            await timed("the next call", session.call_tool("ping", {}, read_timeout_seconds=5))

    print("\na client that reconnects once when a call fails")
    await timed("ping through a fresh session", call_with_reconnect("ping"))


async def call_with_reconnect(name: str, arguments: dict | None = None, attempts: int = 2):
    for attempt in range(attempts):
        try:
            async with stdio_client(PARAMS) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    return await session.call_tool(name, arguments or {}, read_timeout_seconds=5)
        except Exception:                                                # noqa: BLE001
            if attempt == attempts - 1:
                raise


asyncio.run(main())
