# timeout: 1200
# Before optimising anything: what does one request actually cost, and where?

import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST, rate                    # noqa: E402
from clarity.platform.instrumented import (TracedClient,       # noqa: E402
                                           traced_tools)
from clarity.platform.tracing import span, start, waterfall    # noqa: E402
from clarity.v0_6.retrieve import Retriever                    # noqa: E402
from clarity.v0_7.warehouse import Warehouse                   # noqa: E402
from clarity.v0_8.tools import build_tools                     # noqa: E402
from clarity.v0_9.agent import Agent, Budget                   # noqa: E402
from meridian_index import load_index                          # noqa: E402

key = MODEL_FAST.upper().replace("-", "_").replace(".", "_")
placeholder = rate(MODEL_FAST) is None
if placeholder:
    os.environ[f"RATE_{key}_INPUT"] = "0.15"
    os.environ[f"RATE_{key}_OUTPUT"] = "0.60"

QUESTION = "What was revenue in 2024 Q3, and what caused the Midwest decline?"
SYSTEM = "You are an analyst for Meridian. Use tools for every fact."

chunks, vectors = load_index()
sink = start()
client = TracedClient()
tools = traced_tools(build_tools(Retriever(chunks, vectors, client=client),
                                 Warehouse(client=client)))
with span("clarity.request"):
    Agent(tools, client=client, budget=Budget(steps=6)).run(QUESTION, system=SYSTEM)

rows = sink.rows()
calls = [r for r in rows if r["name"].startswith("model.")]
total_cost = sum(r["cost"] for r in rows)
total_in = sum(r["tokens_in"] for r in rows)
total_out = sum(r["tokens_out"] for r in rows)

print(f"Q: {QUESTION}\n")
print(f"  {'span':<23}{'':<18}{'time':>8} {'tok in/out':<8} {'cost':>8}")
print(waterfall(rows))

print(f"\n  {total_in:,} tokens in, {total_out:,} out, ${total_cost:.5f} total\n")

by_name = defaultdict(lambda: [0, 0.0, 0, 0])
for row in calls:
    entry = by_name[row["name"]]
    entry[0] += 1
    entry[1] += row["cost"]
    entry[2] += row["tokens_in"]
    entry[3] += row["tokens_out"]

print(f"  {'':<16}{'calls':>7}{'in':>9}{'out':>7}{'cost':>10}{'share':>8}")
for name, (count, cost, tin, tout) in sorted(by_name.items(),
                                             key=lambda kv: -kv[1][1]):
    print(f"  {name:<16}{count:>7}{tin:>9,}{tout:>7,}{cost:>10.5f}"
          f"{cost / total_cost:>8.0%}")

json.dump({"total_cost": total_cost, "tokens_in": total_in, "tokens_out": total_out,
           "by_name": {k: {"calls": v[0], "cost": v[1], "in": v[2], "out": v[3]}
                       for k, v in by_name.items()},
           "placeholder": placeholder},
          open("code/28/_breakdown.json", "w"), indent=1)

print()
print(f"  input tokens are {total_in / (total_in + total_out):.0%} of the volume and "
      f"cost less per token,")
print(f"  output tokens are {total_out / (total_in + total_out):.0%} of the volume and "
      f"cost more")
print()
if placeholder:
    print("  (placeholder rates — this book prints no price list. Set RATE_* in .env")
    print("   from your provider's page and every figure below becomes yours.)")
    print()
print("Three things in that breakdown decide every optimisation in this chapter.")
print()
print("The input is mostly the same text, over and over. Each model call re-sends the")
print("system prompt, the tool schemas, and every message so far — which is why a")
print("conversation gets more expensive per turn, and why §28.2's provider-side")
print("prompt caching exists.")
print()
print("Most of the calls are not the ones you think about. An agent's visible work is")
print("one question and one answer; the trace shows a planner, an embedding, and a")
print("call per step. The step you never think about is often the one to remove.")
print()
print("And the cheapest token is the one you do not generate. Output is a minority of")
print("the volume at several times the price, which makes §28.2 — asking for shorter")
print("answers — the most overlooked lever in the field.")
