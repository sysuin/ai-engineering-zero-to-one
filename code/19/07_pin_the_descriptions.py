# timeout: 300
# A description you reviewed is not the description you will be sent. Pin the tool
# list by hash when you approve a server, and check it before every use.

import asyncio
import hashlib
import json
import sys
import textwrap

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(command=sys.executable, args=["code/19/_rug_pull_server.py"])


def fingerprint(tools) -> str:
    """Everything a model reads about the tools: names, descriptions and schemas."""
    listed = sorted((t.name, t.description, json.dumps(t.input_schema, sort_keys=True))
                    for t in tools)
    return hashlib.sha256(json.dumps(listed).encode()).hexdigest()[:12]


async def main() -> None:
    notices = []

    async def on_message(message) -> None:
        notices.append(type(message).__name__)

    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write, message_handler=on_message) as session:
            await session.initialize()
            reviewed = (await session.list_tools()).tools
            pinned = fingerprint(reviewed)
            print(f"at review      {pinned}")
            print(f"               {reviewed[0].description}")

            for attempt in (1, 2):
                current = (await session.list_tools()).tools
                seen = fingerprint(current)
                if seen != pinned:
                    print(f"before call {attempt}  {seen}")
                    print(textwrap.fill(current[0].description, 72,
                                        initial_indent=" " * 15, subsequent_indent=" " * 15))
                    print("               no longer what was reviewed: refusing to call it")
                    break
                result = await session.call_tool("convert", {"amount": 100, "rate": 0.79})
                print(f"call {attempt}         {seen}  matches; converted -> "
                      f"{result.content[0].text}")

    print(f"\nnotifications the server sent about the change: {len(notices)}")
asyncio.run(main())
