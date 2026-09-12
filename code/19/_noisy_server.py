# skip
"""
A Clarity server with one line of debugging in it. Started by 02_break_the_wire.py.

    stderr    the line goes where it belongs
    stdout    the line goes on the protocol, before serving starts
    midframe  a tool writes to stdout while answering, with no newline

Nothing else differs between the three.
"""
import sys

sys.path.insert(0, "code")
from clarity.v0_11.mcp_server import build              # noqa: E402

WHERE = sys.argv[1] if len(sys.argv) > 1 else "stderr"
server = build()

if WHERE == "midframe":
    # The worst version: not a line of its own, but a fragment with no newline,
    # written while a tool is running. It lands immediately before the response.
    @server.tool(description="Return a fixed number.")
    def chatty() -> int:
        sys.stdout.write("[debug] computing")
        sys.stdout.flush()
        return 42
else:
    print("[debug] index loaded, serving",
          file=sys.stdout if WHERE == "stdout" else sys.stderr, flush=True)

    @server.tool(description="Return a fixed number.")
    def chatty() -> int:
        return 42

server.run(transport="stdio")
