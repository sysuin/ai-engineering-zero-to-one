# skip
"""A server whose tools misbehave on request: one hangs, one kills the server."""
import os
import time

from mcp.server.mcpserver import MCPServer

server = MCPServer(name="fragile", log_level="ERROR")


@server.tool(description="Reply at once.")
def ping() -> str:
    return "pong"


@server.tool(description="Take this many seconds to reply.")
def slow(seconds: float) -> str:
    time.sleep(seconds)
    return "finally"


@server.tool(description="Stop the server process without replying.")
def crash() -> str:
    os._exit(3)


if __name__ == "__main__":
    server.run(transport="stdio")
