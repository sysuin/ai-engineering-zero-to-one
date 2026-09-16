# timeout: 2400
# A step budget chosen from how long runs actually take. Four questions, five runs each, with a
# ceiling high enough that nothing is cut short — then the ceiling you would set from that.

import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

chunks, vectors = load_index()
agent = Agent(build_tools(Retriever(chunks, vectors), Warehouse()),
              budget=Budget(steps=20, seconds=400, tokens=200_000))
SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool. When you have enough, answer.")
QUESTIONS = [
    "Margin fell in 2025 Q1. Find out by how much, and what the company says caused it.",
    "Something changed in the Midwest during 2024. Work out what, and quantify it.",
    "Are customers complaining about a product problem? If so, which supplier, and how big is it?",
    "Which supplier represents the biggest commercial risk, and why?",
]
RUNS = 5

with ThreadPoolExecutor(max_workers=10) as pool:
    runs = list(pool.map(lambda q: (q, agent.run(q, SYSTEM)), [q for q in QUESTIONS for _ in range(RUNS)]))

print(f"{len(QUESTIONS)} questions x {RUNS} runs, step ceiling 20\n")
steps_all = []
for question in QUESTIONS:
    mine = [r for q, r in runs if q == question]
    steps = sorted(len(r.steps) for r in mine)
    steps_all += steps
    cut = sum(r.stopped_because != "answered" for r in mine)
    print(f"  {question[:58]:59} steps {' '.join(map(str, steps)):14} hit the ceiling {cut}")

steps_all.sort()
p90 = steps_all[int(0.9 * (len(steps_all) - 1))]
print(f"\nall {len(steps_all)} runs: median {statistics.median(steps_all):.0f} steps, "
      f"90th percentile {p90}, longest {steps_all[-1]}")
for ceiling in (4, 6, 8, p90 + 2):
    print(f"  a ceiling of {ceiling:>2} would have been reached by {sum(s >= ceiling for s in steps_all):>2} of {len(steps_all)} runs")
