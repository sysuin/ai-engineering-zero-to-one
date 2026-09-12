# timeout: 900
# One stray print(), and what actually happens — which is not the folklore.

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# The client logs parse failures. Capture them rather than letting them scroll past,
# because in this experiment they are the finding.
PARSE_ERRORS: list[str] = []


class Collect(logging.Handler):
    def emit(self, record): PARSE_ERRORS.append(record.getMessage())


logging.getLogger("mcp.client.stdio").addHandler(Collect())
logging.getLogger("mcp.client.stdio").propagate = False

HELLO = {"jsonrpc": "2.0", "id": 1, "method": "initialize",
         "params": {"protocolVersion": "2025-11-25", "capabilities": {},
                    "clientInfo": {"name": "by-hand", "version": "0"}}}


def raw_first_line(where: str) -> str:
    """Speak the protocol by hand, so the wire is visible rather than described."""
    server = subprocess.Popen([sys.executable, "code/19/_noisy_server.py", where],
                              stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                              stderr=subprocess.DEVNULL, text=True, bufsize=1)
    server.stdin.write(json.dumps(HELLO) + "\n")
    server.stdin.flush()
    line = server.stdout.readline()
    server.kill()
    return line


print("=== 1. is the junk really on the wire? ===\n")
for where in ("stderr", "stdout"):
    line = raw_first_line(where)
    print(f"debug line -> sys.{where}")
    print(f"  server's first stdout line: {line.strip()[:74]}")
    try:
        json.loads(line)
        print("  parses as a JSON-RPC frame")
    except json.JSONDecodeError as error:
        print(f"  json.JSONDecodeError: {error}")
    print()


async def connect(where: str, call_tool: bool) -> str:
    params = StdioServerParameters(command=sys.executable,
                                   args=["code/19/_noisy_server.py", where])
    with open("data/meridian/noisy.log", "w") as errlog:
        try:
            async with stdio_client(params, errlog=errlog) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=15)
                    if not call_tool:
                        listed = await session.list_tools()
                        return f"connected, {len(listed.tools)} tools listed"
                    result = await asyncio.wait_for(
                        session.call_tool("chatty", {}), timeout=15)
                    return f"tool returned {result.content[0].text}"
        except BaseException as error:                       # noqa: BLE001
            while getattr(error, "exceptions", None):
                error = error.exceptions[0]
            if isinstance(error, (asyncio.TimeoutError, TimeoutError)):
                return "TimeoutError — still waiting for a frame that never came"
            return f"{type(error).__name__}: {str(error).splitlines()[0][:60]}"


print("=== 2. what does a real client do with it? ===\n")
for where in ("stderr", "stdout"):
    PARSE_ERRORS.clear()
    outcome = asyncio.run(connect(where, call_tool=False))
    print(f"debug line -> sys.{where:<7} {outcome}")
    print(f"                          {len(PARSE_ERRORS)} parse errors logged")

print()
print("Read that twice. The client with a corrupted first line still connected.")
print("It logged a parse failure, discarded the line, and carried on — so the")
print("folklore version of this bug, 'one print breaks your server', is out of date")
print("for this client. The junk is real; the client is tolerant of it.")
print()
print("Which makes the bug worse, not better. It is now silent.")

print("\n=== 3. what happens to a print inside a tool ===\n")
print("A tool that writes to stdout *while answering*, with no newline, so the")
print("fragment should land immediately in front of the response frame:\n")
PARSE_ERRORS.clear()
outcome = asyncio.run(connect("midframe", call_tool=True))
print(f"  tools/call chatty -> {outcome}")
print(f"  parse errors logged: {len(PARSE_ERRORS)}")
print(f"  the server's stderr afterwards: "
      f"{Path('data/meridian/noisy.log').read_text().strip()[:40]!r}")
print()
print("It worked, and the debug fragment came out on stderr — a stream the tool")
print("never named. That is not luck. While it is serving, this SDK points file")
print("descriptor 1 at stderr and descriptor 0 at the null device, so a stray write")
print("from a handler, a library, or a child process physically cannot reach the")
print("wire. It is restored when the server stops.")
print()
print("So both halves of the folklore are wrong for this SDK, in opposite")
print("directions, and the true rule is narrower and more useful:")
print()
print("  * during serving   the SDK protects the wire for you")
print("  * before serving   nothing does, and section 1 is what you get")
print()
print("Everything printed before run() is on the protocol: a module-level print, an")
print("imported library's banner, a progress bar, a deprecation warning. That is the")
print("window, and it is exactly where people put their debugging.")
print()
print("Do not rely on the protection either. It is this SDK, this version, this")
print("transport; a client speaking to you over HTTP has no fd 1 to rescue. Give the")
print("server a log() that writes to stderr, never allow a bare print() into the")
print("file, and you are correct on every version of every transport.")
