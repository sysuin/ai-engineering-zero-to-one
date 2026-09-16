# timeout: 600
# Five ways to be wrong, and which of them the model is supposed to read.

import asyncio
import json
import subprocess
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

PARAMS = StdioServerParameters(command=sys.executable,
                               args=["code/clarity/v0_11/mcp_server.py"])
CASES = [
    ("a question the warehouse cannot answer",
     "query_warehouse", {"question": "how many staff work in Ohio"}),
    ("a tool that does not exist",
     "delete_everything", {}),
    ("a required argument left out",
     "query_warehouse", {}),
    ("an argument of the wrong type",
     "search_documents", {"query": "revenue", "limit": "lots"}),
]


async def through_the_client() -> None:
    with open("data/meridian/server.log", "w") as errlog:
        async with stdio_client(PARAMS, errlog=errlog) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                for label, tool, arguments in CASES:
                    result = await session.call_tool(tool, arguments)
                    text = result.content[0].text if result.content else ""
                    print(f"{label}")
                    print(f"  {tool}({json.dumps(arguments)})")
                    print(f"  is_error={str(result.is_error):<5} "
                          f"{text.strip().splitlines()[0][:60]}")
                    print()


def by_hand(method: str) -> dict:
    """Ask for a method that does not exist, straight down the pipe."""
    hello = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                        "clientInfo": {"name": "by-hand", "version": "0"}}}
    server = subprocess.Popen([sys.executable, "code/clarity/v0_11/mcp_server.py"],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, bufsize=1)
    for frame in (hello, {"jsonrpc": "2.0", "method": "notifications/initialized"},
                  {"jsonrpc": "2.0", "id": 9, "method": method, "params": {}}):
        server.stdin.write(json.dumps(frame) + "\n")
        server.stdin.flush()
        if "id" in frame and frame["id"] == 1:
            server.stdout.readline()
    reply = json.loads(server.stdout.readline())
    server.kill()
    return reply


asyncio.run(through_the_client())
print("Every one of those came back as a tool error, including two that are not the")
print("model's fault at all: a tool that does not exist, and arguments that do not")
print("type-check. This SDK's default is to hand almost everything to the model.\n")

print("A protocol error looks different, and you have to go off the tool path to")
print("provoke one:\n")
reply = by_hand("nonsense/method")
print(f"  {json.dumps(reply)[:76]}")
print(f"  no 'result', no content, code {reply['error']['code']}\n")

print("The distinction is not cosmetic. It decides who is expected to act.")
print()
print("A tool error is a result. It travels in `content`, a model reads it, and the")
print("run continues. Clarity's warehouse refusal is a paragraph telling the model")
print("what the data does hold — Chapter 16 measured terse errors leading to a right")
print("call 3 times in 24, and instructive ones 21.")
print()
print("A protocol error is a bug. No content, no model, nothing downstream that can")
print("recover. It should page somebody.")
print()
print("Which means the line is not where you would put it. 'Unknown tool' is a")
print("client bug being reported to a model as advice, and a validation failure is a")
print("schema disagreement being handed to the one participant who cannot fix it.")
print("If you want your outages to reach you rather than politely confuse a model,")
print("draw that boundary yourself — the protocol will not draw it for you.")
