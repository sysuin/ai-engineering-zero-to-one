# timeout: 2400
# Five failure modes that are widely described. Which of them actually happen here?
# Each is provoked three times and classified by code, so the verdict is the data's.

import json
import re
import sqlite3
import sys
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import Tool, build_tools         # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

RUNS = 3
chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())
SYSTEM = "You are an analyst for Meridian. Use tools for every fact."
client = OpenAI()
report = {}


def runs(question: str, budget: Budget, toolset=tools):
    return [Agent(toolset, budget=budget).run(question, SYSTEM) for _ in range(RUNS)]


def verdict(label: str, occurred: int, detail: str):
    report[label] = {"occurred": occurred, "runs": RUNS}
    print(f"   -> occurred in {occurred} of {RUNS} runs. {detail}\n")


# ---------------------------------------------------------------- 1. looping
print("1. Looping — searching again and again for something that is not there\n")


def never_finds(query: str, limit: int = 5) -> list:
    return []                                   # a search that finds nothing, ever


empty_search = [Tool("search_documents", "Search Meridian's documents for a passage.",
                     {"type": "object",
                      "properties": {"query": {"type": "string"},
                                     "limit": {"type": "integer", "default": 5}},
                      "required": ["query"], "additionalProperties": False},
                     never_finds)]
looped = 0
for run in runs("Find Meridian's employee headcount.", Budget(steps=8), empty_search):
    signatures = [json.dumps(s.arguments, sort_keys=True) for s in run.steps]
    exact = len(signatures) - len(set(signatures))
    looped += "exhausted" in run.stopped_because
    print(f"   {len(run.steps)} searches, {exact} exact repeats, "
          f"stopped: {run.stopped_because}")
verdict("looping", looped, "Counted as a loop: still searching when the budget of 8 ran out.")

# ---------------------------------------------------------------- 2. budget exhaustion
print("2. Budget exhaustion — and what comes back when it happens\n")
exhausted, empty = 0, 0
for run in runs("Compare 2024 Q3 and 2025 Q1 on revenue, margin and the stated causes, "
                "and say which quarter was healthier.", Budget(steps=2)):
    exhausted += "exhausted" in run.stopped_because
    empty += not run.answer.strip()
    print(f"   {run.stopped_because}; answer of {len(run.answer):,} characters: "
          f"{' '.join(run.answer.split())[:70]}…")
verdict("budget", exhausted, f"Runs that came back with an empty answer: {empty}.")

# ---------------------------------------------------------------- 3. thrashing
print("3. Tool thrashing — swapping tools instead of thinking\n")
thrashed = 0
for run in runs("What is the average tenure of Meridian's warehouse staff?", Budget(steps=8)):
    route = run.tools_used
    switches = sum(a != b for a, b in zip(route, route[1:]))
    thrashed += switches >= 3
    print(f"   {len(route)} steps, {switches} switches of tool: {' -> '.join(route)}")
verdict("thrashing", thrashed, "Counted as thrashing: three or more switches of tool.")

# ---------------------------------------------------------------- 4. false premise
print("4. Confident wrongness — building on a false premise\n")


class Premise(BaseModel):
    treats_premise_as_true: bool
    says_premise_is_unsupported: bool


def judge(answer: str) -> Premise:
    return client.beta.chat.completions.parse(
        model=MODEL_FAST, temperature=0, response_format=Premise,
        messages=[{"role": "user", "content":
                   "The question claimed Meridian opened a fourth distribution centre in "
                   "2025. That is false. Read the answer below. Does it treat the claim "
                   "as true (reporting on the centre's performance)? Does it say the "
                   f"claim could not be confirmed or is not supported?\n\n{answer}"}]
    ).choices[0].message.parsed


believed, challenged = 0, 0
for run in runs("Meridian opened a fourth distribution centre in 2025. How has it "
                "performed?", Budget(steps=6)):
    p = judge(run.answer)
    believed += p.treats_premise_as_true
    challenged += p.says_premise_is_unsupported
    print(f"   {len(run.steps)} steps; built on the premise: {p.treats_premise_as_true}; "
          f"said it was unsupported: {p.says_premise_is_unsupported}")
verdict("false_premise", believed, f"Said plainly that the premise was unsupported: "
        f"{challenged} of {RUNS}. (Both judgements made by a model.)")

# ---------------------------------------------------------------- 5. drift
print("5. Drift — answering a nearby question\n")
con = sqlite3.connect("file:data/meridian/warehouse/meridian.db?mode=ro", uri=True)
margins = con.execute("""SELECT region, ROUND(100.0 * SUM(gross_profit) / SUM(revenue), 1)
                         FROM v_sales WHERE year = 2024 AND quarter = 4
                         GROUP BY region ORDER BY 2""").fetchall()
(worst, low), (best, high) = margins[0], margins[-1]
gap = round(high - low, 1)
print(f"   truth: worst {worst} {low}%, best {best} {high}%, gap {gap} points")
drifted = 0
for run in runs("Which region had the worst gross margin in 2024 Q4, and by how much "
                "did it trail the best?", Budget(steps=6)):
    numbers = {float(n) for n in re.findall(r"\d+\.\d+", run.answer)}
    both = worst in run.answer and gap in numbers
    drifted += not both
    print(f"   names {worst}: {worst in run.answer}; states the {gap}-point gap: "
          f"{gap in numbers}; route: {' -> '.join(run.tools_used)}")
verdict("drift", drifted, "Counted as drift: either half missing or wrong.")

Path("code/17/_failures.json").write_text(json.dumps(report, indent=2))
seen = [k for k, v in report.items() if v["occurred"]]
print(f"Occurred at least once: {', '.join(seen) or 'none'}.")
print(f"Never occurred in {RUNS} runs: "
      f"{', '.join(k for k in report if k not in seen) or 'none'}.")
