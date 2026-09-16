# timeout: 300
# Two things a server can tell you about a tool beyond its description: the shape of
# its result, which a client can check, and hints about its behaviour, which it cannot.

import asyncio
import json
import sys
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SCRATCH = Path(tempfile.mkdtemp()) / "board-pack.xlsx"
SCRATCH.write_text("the only copy")
PARAMS = StdioServerParameters(command=sys.executable,
                               args=["code/19/_claims_server.py", str(SCRATCH)])


def needs_approval(tool, server_is_trusted: bool) -> bool:
    """Hints from a server you do not control are marketing. Only a trusted one's count."""
    hints = tool.annotations
    if server_is_trusted and hints and hints.read_only_hint:
        return False
    return True


async def main() -> None:
    async with stdio_client(PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = {t.name: t for t in (await session.list_tools()).tools}

            revenue = tools["quarter_revenue"]
            print("quarter_revenue declares an output schema:")
            fields = revenue.output_schema.get("properties", {})
            print("  " + ", ".join(f"{name}: {spec['type']}" for name, spec in fields.items()))
            result = await session.call_tool("quarter_revenue", {"year": 2024, "quarter": 3})
            print(f"  structured result: {result.structured_content}")
            print(f"  text result:       {json.dumps(json.loads(result.content[0].text))}")

            tidy = tools["tidy_up"]
            print(f"\ntidy_up says read_only_hint={tidy.annotations.read_only_hint}, "
                  f"destructive_hint={tidy.annotations.destructive_hint}")
            print(f"  approval if the server is trusted:   {needs_approval(tidy, True)}")
            print(f"  approval if the server is not:       {needs_approval(tidy, False)}")
            print(f"  before the call, the file exists:    {SCRATCH.exists()}")
            await session.call_tool("tidy_up", {})
            print(f"  after the call, the file exists:     {SCRATCH.exists()}")


asyncio.run(main())
