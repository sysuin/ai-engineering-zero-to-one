# timeout: 1200
# The same loop as Chapter 16, with a budget and a trace — and a check of the answer
# against the warehouse, because the trace is only useful if you know what was right.

import re
import sqlite3
import sys

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever                 # noqa: E402
from clarity.v0_7.warehouse import Warehouse                # noqa: E402
from clarity.v0_8.tools import build_tools                  # noqa: E402
from clarity.v0_9.agent import Agent, Budget                # noqa: E402
from meridian_index import load_index                       # noqa: E402

chunks, vectors = load_index()
agent = Agent(build_tools(Retriever(chunks, vectors), Warehouse()),
              budget=Budget(steps=8, seconds=120))

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool. When you have enough, answer and "
          "stop.")

QUESTION = ("Margin fell in 2025 Q1. Find out by how much, and what the company says "
            "caused it.")

run = agent.run(QUESTION, SYSTEM)

print(f"Q: {QUESTION}\n")
for step in run.steps:
    mark = "FAILED" if step.failed else "ok"
    print(f"  {step.n}. {mark:6} {step.tool:18} {step.seconds:5.2f}s  "
          f"{str(step.arguments)[:50]}")
    print(f"          -> {step.result[:60]}")

print(f"\nstopped because: {run.stopped_because}")
print(f"{len(run.steps)} steps, {run.tokens:,} tokens, {run.seconds:.1f}s")
print(f"\nA: {run.answer[:520]}")

# ---------------------------------------------------------------- check it
con = sqlite3.connect("file:data/meridian/warehouse/meridian.db?mode=ro", uri=True)
margin = dict(((y, q), m) for y, q, m in con.execute(
    """SELECT year, quarter, ROUND(100.0 * SUM(gross_profit) / SUM(revenue), 1)
       FROM v_sales WHERE year IN (2024, 2025) GROUP BY year, quarter"""))
before, after = margin[(2024, 4)], margin[(2025, 1)]
fall = round(before - after, 1)
numbers = {float(n) for n in re.findall(r"\d+\.\d", run.answer)}

print("\nChecked against the warehouse:")
print(f"  2024 Q4 margin {before}%, 2025 Q1 {after}%: a fall of {fall} points")
print(f"  the answer states the 2025 Q1 margin:      {'yes' if after in numbers else 'no'}")
print(f"  the answer states the size of the fall:    "
      f"{'yes' if fall in numbers or before in numbers else 'no'}")
failed = [s.n for s in run.steps if s.failed]
if failed:
    after_failure = [s.tool for s in run.steps if s.n == failed[0] + 1]
    print(f"  step {failed[0]} failed; the next step used "
          f"{after_failure[0] if after_failure else 'nothing'}")
