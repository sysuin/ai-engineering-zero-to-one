# timeout: 600
# Start the server as a separate process and talk to it over stdio.

import asyncio
import json
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(command=sys.executable,
                               args=["code/clarity/v0_11/mcp_server.py"])
STDERR = Path("data/meridian/server.log")


async def describe(session: ClientSession) -> None:
    info = await session.initialize()
    print(f"connected to {info.server_info.name} v{info.server_info.version}")
    print(f"protocol     {info.protocol_version}")
    print(f"offers       {', '.join(k for k, v in info.capabilities if v)}\n")

    listed = await session.list_tools()
    print(f"tools/list -> {len(listed.tools)} tools")
    for tool in listed.tools:
        required = tool.input_schema.get("required", [])
        # A trailing ? marks an optional parameter. The client learned all of this
        # from the server; none of it is written down on this side.
        params = ", ".join(f"{n}{'' if n in required else '?'}"
                           for n in tool.input_schema.get("properties", {}))
        print(f"  {tool.name}({params})")
        print(f"      {tool.description.split('.')[0][:66]}.")


async def call(session: ClientSession) -> None:
    print()
    result = await session.call_tool("query_warehouse",
                                     {"question": "revenue in 2024 Q3"})
    figure = json.loads(result.content[0].text)
    print(f"tools/call query_warehouse  -> {figure['rows']}")
    print(f"                               {figure['sql'][:62]}")

    result = await session.call_tool("search_documents",
                                     {"query": "Midwest decline", "limit": 2})
    # A tool that returns a list comes back as one content block per item.
    passages = [json.loads(block.text) for block in result.content]
    print(f"tools/call search_documents -> {len(passages)} passages")
    for passage in passages:
        print(f"                               {passage['source']} / "
              f"{passage['heading']}")


async def main() -> None:
    # Two pipes and a subprocess. The client writes JSON-RPC frames to the server's
    # stdin and reads them from its stdout, and nothing else may travel on either.
    # The server's stderr is a third stream, and everything human goes there.
    with STDERR.open("w") as errlog:
        async with stdio_client(PARAMS, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await describe(session)
                await call(session)

    print(f"\nserver stderr ({STDERR}):")
    for line in STDERR.read_text().splitlines()[:2]:
        print(f"  {line}")
    print()
    print("Nothing in this file knows what Clarity is. It asked a process what it")
    print("could do, was told, and called one of the answers. The client was written")
    print("before the server existed, and that indirection is the entire point.")


asyncio.run(main())
