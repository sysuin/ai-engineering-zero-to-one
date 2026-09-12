# timeout: 2400
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# The answer was right. Was the route?

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.runner import correct, load                  # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly.")

# The first version of this specification said: a case whose gold answer came from a
# document must be answered with search_documents. That is the obvious rule and it is
# wrong, for a reason the output below makes plain.
NAIVE_TOOL = {"warehouse": "query_warehouse", "document": "search_documents"}

# The rule that survives: every answer must come from *some* source. Clarity may
# reach a figure through the warehouse or through a document — both are sourced —
# but an answer produced with no tool call at all is the model remembering, and that
# is the trajectory failure worth failing a build over.
SOURCES = {"query_warehouse", "search_documents"}

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())
cases = [c for c in load() if c["kind"] in NAIVE_TOOL][:70]


def run(case: dict) -> dict:
    result = Agent(tools, budget=Budget(steps=6)).run(case["question"],
                                                      system=SYSTEM)
    used = [s.tool for s in result.steps if s.tool]
    return {"id": case["id"], "kind": case["kind"], "question": case["question"],
            "outcome": correct(case, result.answer), "tools": used,
            "expected_tool": NAIVE_TOOL[case["kind"]],
            "took_expected": NAIVE_TOOL[case["kind"]] in used,
            "sourced": bool(SOURCES & set(used)),
            "steps": len(result.steps), "tokens": result.tokens}


with ThreadPoolExecutor(max_workers=8) as pool:
    rows = list(pool.map(run, cases))
json.dump(rows, open("code/22/_trajectory.json", "w"), indent=1)

quadrant = Counter((r["outcome"], r["took_expected"]) for r in rows)
print(f"{len(rows)} questions, judged twice: on the answer, and on the route.\n")
print("First specification: the tool must match where the gold answer came from.\n")
print(f"  {'':<30}{'right route':>13}{'wrong route':>13}")
print(f"  {'right answer':<30}{quadrant[(True, True)]:>13}"
      f"{quadrant[(True, False)]:>13}")
print(f"  {'wrong answer':<30}{quadrant[(False, True)]:>13}"
      f"{quadrant[(False, False)]:>13}")

outcome_only = sum(r["outcome"] for r in rows)
print()
print(f"  outcome evaluation alone passes  {outcome_only} of {len(rows)}")
print(f"  adding that trajectory rule      {quadrant[(True, True)]} of {len(rows)}")

off_spec = [r for r in rows if r["outcome"] and not r["took_expected"]]
print()
print(f"Before blaming the system for those {len(off_spec)}, read three of them:\n")
for row in off_spec[:3]:
    print(f"  {row['question'][:56]}")
    print(f"    spec wanted {row['expected_tool']}, it used "
          f"{', '.join(row['tools']) or 'no tools at all'}")
print()
print("The specification is wrong. Those cases are labelled `document` because their")
print("gold answer was verified against a document span — that is a statement about")
print("provenance, not about which tool ought to answer. Meridian's revenue is in the")
print("warehouse *and* in the review, and going to the warehouse for a figure is the")
print("better route, not a deviation.")
print()
print("This is the failure mode of trajectory evaluation in one paragraph. A")
print("trajectory spec is a claim about how the system should work, written by")
print("somebody who may be wrong, and when it disagrees with the system the burden of")
print("proof is genuinely on both.")

sourced = sum(r["sourced"] for r in rows)
unsourced = [r for r in rows if not r["sourced"]]
print()
print("Second specification, and the one worth keeping: every answer must come from")
print("some tool. Either source is fine; no source is not.\n")
print(f"  answers with at least one source call   {sourced} of {len(rows)}")
print(f"  answers produced from memory alone      {len(unsourced)} of {len(rows)}")
for row in unsourced[:2]:
    print(f"    {row['question'][:56]}  ->  "
          f"{', '.join(row['tools']) or 'no tools'}")

print()
print("That rule is narrow enough to be right and cheap enough to run on every case.")
print("It also generalises, which the first one did not: it is a property of the run")
print("rather than a prediction about it.")
print()
print("Where trajectory evaluation genuinely earns its keep is agents that write,")
print("spend, or send. 'Got the right answer' is not a sufficient account of a run")
print("that also despatched two emails on the way, and no outcome check will ever")
print("mention them.")
