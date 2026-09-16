# timeout: 1800
# What does an agent do when a search finds nothing? The twenty unanswerable questions of the
# golden set, with a search that finds nothing, reported three ways: as an empty list, as a terse
# error, and as an error that invites another try. The warehouse tool is the real one.

import dataclasses
import json
import re
import statistics
import textwrap
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.runner import abstained, load                 # noqa: E402
from clarity.v0_7.warehouse import Warehouse                     # noqa: E402
from clarity.v0_8.tools import ToolError, build_tools            # noqa: E402
from clarity.v0_9.agent import Agent, Budget                     # noqa: E402

RUNS, STEPS = 2, 8
SYSTEM = "You are an analyst for Meridian. Use tools for every fact."
cases = [c for c in load() if c["kind"] == "unanswerable"]


class FindsNothing:
    def search(self, query, k=5):
        return []


def reported_as(style: str):
    tools = build_tools(FindsNothing(), Warehouse())
    if style == "an empty list":
        return tools

    def search(**arguments):
        if style == "a terse error":
            raise ToolError("Search failed.")
        raise ToolError("No passages matched that query. Try different words.")
    return [dataclasses.replace(t, run=search) if t.name == "search_documents" else t
            for t in tools]


STYLES = ("an empty list", "a terse error", "an error inviting a retry")
FAILURE = re.compile(r"right now|\bfail|couldn.t retrieve|could not retrieve|unavailable", re.I)
print(f"{len(cases)} unanswerable questions x {RUNS} runs; the search finds nothing; "
      f"step budget {STEPS}\n")
print(f"  {'nothing reported as':<26}{'searches':>9}{'budget out':>12}{'not found':>11}"
      f"{'broken':>8}")
saved, shown = [], {}
for style in STYLES:
    agent = Agent(reported_as(style), budget=Budget(steps=STEPS, seconds=180))
    jobs = [c for c in cases for _ in range(RUNS)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        runs = list(pool.map(lambda c: agent.run(c["question"], SYSTEM), jobs))
    searches = statistics.mean(r.tools_used.count("search_documents") for r in runs)
    steps = statistics.mean(len(r.steps) for r in runs)
    out = sum("exhausted" in r.stopped_because for r in runs)
    said_so = sum(abstained(r.answer) for r in runs)
    broken = sum(bool(FAILURE.search(r.answer)) for r in runs)
    print(f"  {style:<26}{searches:>9.1f}{out:>9}/{len(runs)}{said_so:>8}/{len(runs)}"
          f"{broken:>5}/{len(runs)}")
    saved += [{"style": style, "question": c["question"], "answer": r.answer,
               "abstained": abstained(r.answer)} for c, r in zip(jobs, runs)]
    shown[style] = runs[0].answer
print("\n  searches = mean per run; not found = said so, by Clarity's refusal list;")
print("  broken = said a tool had failed, or it could not be done right now")
for style, answer in shown.items():
    print(f"\n  {style}, for example:")
    print("    " + "\n    ".join(textwrap.wrap(" ".join(answer.split()), 76)[:3]))
json.dump(saved, open("code/16/_empty_as_error.json", "w"), indent=2)
