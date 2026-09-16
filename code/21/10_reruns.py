# timeout: 3600
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# The same system, the same 120 cases, run five times. How much of a score's uncertainty comes
# from which cases were chosen, and how much from the system not giving the same answer twice?

import json
import math
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "code")
from clarity.evals.runner import correct, load                  # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly rather than "
          "guessing.")
RUNS = 5

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())
cases = load()


def graded(case: dict) -> bool:
    run = Agent(tools, budget=Budget(steps=6)).run(case["question"], system=SYSTEM)
    return correct(case, run.answer)


results = np.zeros((RUNS, len(cases)), dtype=bool)
for r in range(RUNS):
    with ThreadPoolExecutor(max_workers=8) as pool:
        results[r] = list(pool.map(graded, cases))

scores = results.mean(axis=1)
per_case = results.mean(axis=0)
flipped = (per_case > 0) & (per_case < 1)
p = scores.mean()
print(f"{len(cases)} cases, {RUNS} runs of the same system\n")
print("  score by run      " + "  ".join(f"{s:.0%}" for s in scores))
print(f"  range             {scores.min():.0%} to {scores.max():.0%}")
print(f"  cases that gave a different verdict in at least one run: {flipped.sum()}")
kinds = Counter(c["kind"] for c, f in zip(cases, flipped) if f)
print("    by kind: " + ", ".join(f"{k} {n}" for k, n in kinds.most_common()))

print("\nwhere the uncertainty in one run's score comes from")
between_runs = scores.std(ddof=1)
case_sampling = math.sqrt(p * (1 - p) / len(cases))
print(f"  {f'which cases were chosen (binomial, n = {len(cases)})':50} ± {case_sampling:.1%}")
print(f"  {'the system answering differently (sd across runs)':50} ± {between_runs:.1%}")

print("\nthree ways to summarise the runs")
print(f"  pass@1  (average share correct per run)          {per_case.mean():.0%}")
print(f"  pass@{RUNS}  (correct in at least one run)            {(per_case > 0).mean():.0%}")
print(f"  pass^{RUNS}  (correct in every run)                   {(per_case == 1).mean():.0%}")

json.dump({"scores": scores.tolist(), "per_case": per_case.tolist(),
           "ids": [c["id"] for c in cases]}, open("code/21/_reruns.json", "w"), indent=1)
