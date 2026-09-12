# timeout: 900
# Uses clarity/platform/tracing.py and instrumented.py — the whole of v0.15's
# observability layer. Nothing inside Clarity itself changed to produce this.
# One question, one trace, every span labelled with time and money.

import json
import os
import sys

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST, rate                     # noqa: E402
from clarity.platform.instrumented import TracedClient, traced_tools  # noqa: E402
from clarity.platform.tracing import span, start, waterfall     # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

# Rates come from .env, and this book does not print a price list. If none are set,
# the placeholder below is used *and said out loud*, because a dashboard with no
# money on it cannot demonstrate the thing this chapter is about.
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
# Traced all the way down: the agent's client, the retriever's, and the warehouse's.
retriever = Retriever(chunks, vectors, client=client)
warehouse = Warehouse(client=client)
tools = traced_tools(build_tools(retriever, warehouse))

with span("clarity.request", **{"clarity.question": QUESTION,
                                "clarity.tenant": "meridian"}):
    Agent(tools, client=client, budget=Budget(steps=6),
          model=MODEL_FAST).run(QUESTION, system=SYSTEM)
deep = list(sink.rows())


def collapse(rows: list[dict]) -> list[dict]:
    """
    The same trace as it would look with shallower instrumentation.

    Drop every span whose parent is a tool call. Nothing is re-run and no timing
    changes — this is one trace viewed twice, which is the only way to compare the two
    honestly. Run the system twice instead and the model takes a different route, and
    you are comparing paths rather than instrumentation.
    """
    tools_ids = {r["span_id"] for r in rows if r["name"].startswith("tool.")}
    return [r for r in rows if r["parent"] not in tools_ids]


shallow = collapse(deep)


def model_share(rows: list[dict]) -> tuple[float, float]:
    total = max(r["ms"] for r in rows)
    named = sum(r["ms"] for r in rows if r["name"].startswith("model."))
    return named, named / total


print(f"Q: {QUESTION}\n")
print("=== what a shallow trace shows ===\n")
print(f"  {'span':<23}{'':<18}{'time':>8} {'tok in/out':<8} {'cost':>8}")
print(waterfall(shallow))
named, share = model_share(shallow)
print(f"\n  {len(shallow)} spans; {named:.0f}ms of {max(r['ms'] for r in shallow):.0f}ms"
      f" sits inside a span named model.* ({share:.0%})")

print("\n=== the same trace, nothing re-run, nothing dropped ===\n")
print(waterfall(deep))
named_deep, share_deep = model_share(deep)
total = max(r["ms"] for r in deep)
cost = sum(r["cost"] for r in deep)
print(f"\n  {len(deep)} spans, {total:.0f}ms end to end, ${cost:.5f}")
print(f"  {named_deep:.0f}ms of it is a model call ({share_deep:.0%})")

hidden = [r for r in deep if r not in shallow]
json.dump({"deep": [{k: v for k, v in r.items() if k != "attributes"} for r in deep],
           "shallow_ids": [r["span_id"] for r in shallow],
           "total_ms": total, "cost": cost,
           "shallow_share": share, "deep_share": share_deep,
           "placeholder": placeholder},
          open("code/23/_trace.json", "w"), indent=1)

if placeholder:
    print()
    print("  (dollar figures use placeholder rates — this book does not print a price")
    print("   list, because one printed with confidence is worse than none. Set RATE_*")
    print("   in .env from the provider's page and the numbers become yours.)")

print()
print("Read the shape first. The tool spans are siblings of the model calls, not")
print("children, because the agent alternates: ask, run, ask again. A waterfall makes")
print("that obvious in a way no log file does, and it is the first thing you want when")
print("a run takes six seconds and nobody can say which part.")
print()
print(f"Then read the two shares. On the shallow view {share:.0%} of the run is model")
print(f"latency and the rest is 'the tools being slow'. On the deep view it is "
      f"{share_deep:.0%},")
print(f"because {len(hidden)} of those spans are model calls that were happening inside "
      f"the tools")
print("all along — the QuerySpec planner from §15.7, and an embedding call for the")
print("query.")
print()
print("Nothing was re-run between those two blocks. It is one trace, filtered.")
print()
print("That is the sharpest thing tracing teaches, and it is not a technique. A trace")
print("is only as deep as your instrumentation, and an uninstrumented call does not")
print("show up as missing — it shows up as its caller being slow. Your first trace")
print("tells you where your instrumentation stops. The second one starts telling you")
print("about the system.")
