# timeout: 1200
# The same agent, the same question, tools reached two different ways.

import json
import sys
import time

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent                     # noqa: E402
from clarity.v0_11.mcp_client import Connection          # noqa: E402
from meridian_index import load_index                    # noqa: E402

RUNS = 5
QUESTION = "What was revenue in 2024 Q3, and what caused the Midwest decline?"
SYSTEM = "You are an analyst for Meridian. Use tools for every fact."

chunks, vectors = load_index()
direct = build_tools(Retriever(chunks, vectors), Warehouse())

started = time.perf_counter()
connection = Connection(sys.executable, ["code/clarity/v0_11/mcp_server.py"],
                        "data/meridian/adapter.log")
proxied = connection.tools()
handshake = time.perf_counter() - started


def measure(tools) -> tuple[list[float], int, int]:
    times, tokens, calls = [], 0, 0
    for _ in range(RUNS):
        started = time.perf_counter()
        run = Agent(tools).run(QUESTION, system=SYSTEM)
        times.append(time.perf_counter() - started)
        tokens += run.tokens
        calls += len([s for s in run.steps if s.tool])
    return sorted(times), tokens // RUNS, calls // RUNS


print(f"starting the server and shaking hands: {handshake:.1f}s, once\n")
print(f"{'':<12}{'median':>9}{'range':>16}{'tokens':>10}{'tool calls':>12}")
results = {}
for label, tools in (("direct", direct), ("over MCP", proxied)):
    times, tokens, calls = measure(tools)
    results[label] = {"median": times[RUNS // 2], "min": times[0], "max": times[-1],
                      "tokens": tokens, "calls": calls}
    print(f"{label:<12}{times[RUNS // 2]:>8.1f}s"
          f"{f'{times[0]:.1f}-{times[-1]:.1f}s':>16}{tokens:>10,}{calls:>12}")
connection.close()

gap = results["over MCP"]["median"] - results["direct"]["median"]
per_call = gap / max(results["over MCP"]["calls"], 1)
spread = max(r["max"] - r["min"] for r in results.values())
results["handshake"] = handshake
json.dump(results, open("code/19/_overhead.json", "w"), indent=1)

print()
print(f"difference in median: {gap:+.2f}s over "
      f"{results['over MCP']['calls']} tool calls, so roughly {per_call:+.2f}s each")
print(f"spread within a single version: {spread:.2f}s")
print(f"-> the per-run difference is "
      f"{'inside' if abs(gap) <= spread else 'outside'} the noise")
print()
print("The token counts are the number that matters and they barely move: the model")
print("is sent the same schemas and gets back the same results, because the adapter")
print("hands the server's own descriptions straight to it. A protocol between your")
print("agent and your tools is invisible to the thing you are paying by the token.")
print()
print("What you do pay is the handshake at the top of this output — a process start,")
print("an index load and an initialize round trip — and you pay it per connection,")
print("not per call. Open the session once and hold it, which is the only reason the")
print("adapter in v0_11/mcp_client.py has a thread in it at all.")
print()
print("So the question is never overhead. It is whether you need the indirection:")
print()
print("  you do not    the tools live in your repository, ship with your code and")
print("                are used by exactly one application")
print("  you do        somebody else's application needs them, or somebody else's")
print("                tools need you, and neither of you wants to be a library")
