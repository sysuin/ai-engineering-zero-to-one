# timeout: 2400
# Forty real requests, and the number a dashboard would have shown you.

import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.runner import load                           # noqa: E402
from clarity.platform.instrumented import TracedClient, traced_tools  # noqa: E402
from clarity.platform.tracing import span, start                # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly.")

# A mix that looks like traffic rather than a benchmark: mostly easy lookups, a few
# that need both halves, a few with no answer at all.
cases = load()
mix = (cases[:14] + [c for c in cases if c["kind"] == "warehouse"][:10] +
       [c for c in cases if c["kind"] == "composite"][:6] +
       [c for c in cases if c["kind"] == "unanswerable"][:10])

chunks, vectors = load_index()
sink = start()


def one(case: dict) -> dict:
    client = TracedClient()
    tools = traced_tools(build_tools(Retriever(chunks, vectors, client=client),
                                     Warehouse(client=client)))
    with span("clarity.request", **{"clarity.kind": case["kind"]}) as current:
        result = Agent(tools, client=client, budget=Budget(steps=6)).run(
            case["question"], system=SYSTEM)
        current.set_attribute("clarity.steps", len(result.steps))
    return {"kind": case["kind"], "tokens": result.tokens,
            "steps": len(result.steps)}


with ThreadPoolExecutor(max_workers=6) as pool:
    meta = list(pool.map(one, mix))

requests = [r for r in sink.rows() if r["name"] == "clarity.request"]
latency = sorted(r["ms"] for r in requests)


def pct(values: list[float], p: float) -> float:
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


mean = statistics.fmean(latency)
stats = {"n": len(latency), "mean": mean, "p50": pct(latency, 50),
         "p90": pct(latency, 90), "p95": pct(latency, 95), "p99": pct(latency, 99),
         "max": latency[-1], "min": latency[0]}

print(f"{len(latency)} requests, run concurrently the way traffic arrives.\n")
for name in ("min", "p50", "mean", "p90", "p95", "p99", "max"):
    value = stats[name]
    bar = "#" * max(1, round(value / stats["max"] * 30))
    marker = "  <- most dashboards" if name == "mean" else ""
    print(f"  {name:<5}{value:>7.0f}ms  {bar:<30}{marker}")

above_mean = sum(1 for v in latency if v > mean)
json.dump({"latency": latency, **stats,
           "kinds": [{"kind": m["kind"], "steps": m["steps"]} for m in meta]},
          open("code/23/_latency.json", "w"), indent=1)

print()
print(f"  requests slower than the mean: {above_mean} of {len(latency)} "
      f"({above_mean / len(latency):.0%})")
print(f"  p99 is {stats['p99'] / stats['p50']:.1f}x the median")
if len(latency) < 100:
    print(f"  (with {len(latency)} requests the p99 is simply the slowest one; "
          f"a p99 needs hundreds of requests to mean anything)")

print()
print("The mean sits where nobody lives. This distribution has a long right tail —")
print("agents that took an extra step, questions that needed two tools instead of")
print("one — and a tail pulls a mean upward without describing anything.")
print()
print(f"Report the mean and you are quoting a figure that "
      f"{1 - above_mean / len(latency):.0%} of requests beat,")
print(f"so most people think you are being pessimistic; meanwhile the "
      f"{above_mean / len(latency):.0%} having a")
print("genuinely bad time are invisible inside it.")
print()
print("Percentiles say something you can act on. p50 is the typical request. p95 is")
print("the one a user notices and mentions. p99 is the one that produces a ticket.")
print()
print("Two rules that follow, and the second one gets broken constantly:")
print()
print("  never average a percentile      the mean of two p95s is not a p95, and a")
print("                                  dashboard that rolls hourly p95s into a")
print("                                  daily one is showing you a number with no")
print("                                  meaning at all")
print("  bucket before you aggregate     a p95 across every request is dominated by")
print("                                  whichever kind is slowest and most common")

by_kind: dict[str, list[float]] = {}
for row, info in zip(requests, meta):
    by_kind.setdefault(info["kind"], []).append(row["ms"])
print()
print("  Which matters here:\n")
for kind, values in sorted(by_kind.items(), key=lambda kv: -pct(sorted(kv[1]), 95)):
    values = sorted(values)
    print(f"    {kind:<14}{len(values):>3} requests   p50 {pct(values, 50):>6.0f}ms   "
          f"p95 {pct(values, 95):>6.0f}ms")
ratios = sorted((pct(sorted(v), 95) for v in by_kind.values()))
print()
print(f"The slowest population's p95 is {ratios[-1] / ratios[0]:.1f}x the fastest "
      f"one's, and a single")
print(f"overall p95 of {stats['p95']:.0f}ms sits between them describing neither. If "
      f"the slow kind is")
print("the one your best customers use, that number is not merely uninformative — it")
print("is reassuring, which is worse.")
