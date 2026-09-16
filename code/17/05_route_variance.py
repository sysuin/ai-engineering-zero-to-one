# timeout: 1800
# An agent is a distribution over routes. The same question, five times, measured.

import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

RUNS = 5
chunks, vectors = load_index()
agent = Agent(build_tools(Retriever(chunks, vectors), Warehouse()),
              budget=Budget(steps=8, seconds=180))
SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool. When you have enough, answer.")
QUESTION = "Something changed in the Midwest during 2024. Work out what, and quantify it."
LETTER = {"search_documents": "S", "query_warehouse": "Q", "arithmetic": "A", "today": "T"}

with ThreadPoolExecutor(max_workers=RUNS) as pool:
    results = list(pool.map(lambda _: agent.run(QUESTION, SYSTEM), range(RUNS)))

print(f"Q: {QUESTION}\n")
print(f"  {'run':>3}  {'route':12} {'steps':>5} {'tokens':>7} {'seconds':>8}  stopped")
for i, run in enumerate(results, 1):
    route = "".join(LETTER.get(t, "?") for t in run.tools_used)
    print(f"  {i:>3}  {route:12} {len(run.steps):>5} {run.tokens:>7,} {run.seconds:>8.1f}"
          f"  {run.stopped_because}")
print("\n  S = search_documents, Q = query_warehouse, A = arithmetic")

tokens = [r.tokens for r in results]
seconds = [r.seconds for r in results]
routes = {tuple(r.tools_used) for r in results}
print(f"\n{len(routes)} distinct routes in {RUNS} runs")
print(f"tokens:  {min(tokens):,} to {max(tokens):,} (median {statistics.median(tokens):,.0f}), "
      f"a spread of {max(tokens) / min(tokens):.1f}x")
print(f"seconds: {min(seconds):.1f} to {max(seconds):.1f}")
print(f"runs that hit a budget: {sum('exhausted' in r.stopped_because for r in results)}"
      f" of {RUNS}")
