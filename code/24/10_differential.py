# timeout: 2400
# Differential debugging: run a flaky question until there are passing and failing runs, then
# compare their traces step by step. The first step where the groups differ is the bug's address.

import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

chunks, vectors = load_index()
agent = Agent(build_tools(Retriever(chunks, vectors), Warehouse()), budget=Budget(steps=8, seconds=240))
SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool. When you have enough, answer.")
QUESTION = "Margin fell in 2025 Q1. Find out by how much, and what the company says caused it."
MUST = ("0.3", "Voss")                                   # the fall in points, and the stated cause
RUNS = 10


def signature(step) -> str:
    """What a step did, with the incidental wording of its arguments removed: which quarter it asked about."""
    if step.tool == "query_warehouse":
        text = step.arguments.get("question", "")
        found = re.search(r"(20\d\d)\s*Q([1-4])|Q([1-4])\s*(20\d\d)", text, re.IGNORECASE)
        if not found:
            return "warehouse: no quarter named"
        year, quarter = (found.group(1), found.group(2)) if found.group(1) else (found.group(4), found.group(3))
        return f"warehouse: {year} Q{quarter}"
    if step.tool == "search_documents":
        return "search"
    return step.tool or "?"


with ThreadPoolExecutor(max_workers=RUNS) as pool:
    runs = list(pool.map(lambda _: agent.run(QUESTION, SYSTEM), range(RUNS)))

passed = [r for r in runs if all(m.lower() in r.answer.lower() for m in MUST)]
failed = [r for r in runs if r not in passed]
print(f"{RUNS} runs: {len(passed)} passed, {len(failed)} failed (an answer must state the 0.3-point fall and Voss)\n")
for label, group in (("passed", passed), ("failed", failed)):
    for r in group:
        print(f"  {label:7} " + " -> ".join(signature(s) for s in r.steps))

print("\nwhat the groups did, step by step (share of runs in each group that took that step)")
depth = max(len(r.steps) for r in runs)
for i in range(depth):
    p = Counter(signature(r.steps[i]) if i < len(r.steps) else "(answered)" for r in passed)
    f = Counter(signature(r.steps[i]) if i < len(r.steps) else "(answered)" for r in failed)
    options = sorted(set(p) | set(f))
    cells = "   ".join(f"{o}: {p[o] / max(1, len(passed)):.0%}/{f[o] / max(1, len(failed)):.0%}" for o in options)
    print(f"  step {i + 1}  {cells}")

ever = lambda group, sig: sum(any(signature(s) == sig for s in r.steps) for r in group)   # noqa: E731
print("\nsteps taken at least once, passed / failed:")
for sig in sorted({signature(s) for r in runs for s in r.steps}):
    print(f"  {sig:22} {ever(passed, sig)}/{len(passed)}   {ever(failed, sig)}/{len(failed)}")


with_q4 = [r for r in runs if any(signature(s) == "warehouse: 2024 Q4" for s in r.steps)]
without = [r for r in runs if r not in with_q4]
rate = lambda group: sum(r in passed for r in group) / len(group) if group else float("nan")   # noqa: E731
print(f"\nruns that asked for 2024 Q4, the quarter before: {len(with_q4)}, passed {rate(with_q4):.0%}")
print(f"runs that never did:                            {len(without)}, passed {rate(without):.0%}")
