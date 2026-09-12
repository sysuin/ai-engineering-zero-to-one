# timeout: 1200
# The same loop as Chapter 16, with a budget and a trace.

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
print(f"\nA: {run.answer[:420]}")

print()
print("Read the trace rather than the answer. The answer is one paragraph; the trace is")
print("the reasoning, and it is the only place a wrong answer can be diagnosed.")
