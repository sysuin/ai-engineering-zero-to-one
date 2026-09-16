# timeout: 2400
# Episodic memory, measured. A note from an earlier run is put in front of the agent, the way a
# memory store retrieves one: accurate, out of date, out of date and labelled as such, and a note
# that remembers the route but keeps no figure.
# The stale figure is simulated — 6% above the warehouse's — and every answer is checked for both.

import json
import re
import statistics
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.runner import load, normalise                 # noqa: E402
from clarity.v0_6.retrieve import Retriever                      # noqa: E402
from clarity.v0_7.warehouse import Warehouse                     # noqa: E402
from clarity.v0_8.tools import build_tools                       # noqa: E402
from clarity.v0_9.agent import Agent, Budget                     # noqa: E402
from clarity.v1_0.clarity import SYSTEM                          # noqa: E402
from meridian_index import load_index                            # noqa: E402

RUNS, DRIFT = 5, 1.06
IDS = ["warehouse-01", "warehouse-02", "warehouse-06", "warehouse-07", "warehouse-08",
       "warehouse-11", "warehouse-13", "warehouse-14", "warehouse-17", "warehouse-18"]
cases = [c for c in load() if c["id"] in IDS]
chunks, vectors = load_index()
agent = Agent(build_tools(Retriever(chunks, vectors), Warehouse()),
              budget=Budget(steps=8, seconds=180))


def figure(text: str, scale: float = 1.0) -> str:
    """The gold answer, optionally drifted, written the way the gold set writes it."""
    value = float(text.replace(",", "")) * scale
    decimals = len(text.split(".")[1]) if "." in text else 0
    return f"{value:,.{decimals}f}"


def forms(text: str) -> list[str]:
    """Ways an answer can state a figure: in full, whole units, or millions to two places."""
    value = float(text.replace(",", ""))
    found = {normalise(text), f"{value:.0f}"}
    if value >= 1_000_000:
        found.add(f"{value / 1e6:.2f}")
    return sorted(found)


def mentions(answer: str, text: str) -> bool:
    return any(form in normalise(answer) for form in forms(text))


def note(case: dict, scale: float) -> str:
    return (f"Notes from earlier runs, retrieved for this question:\n"
            f"- 2025-10-02, asked \"{case['question']}\" and answered "
            f"{figure(case['answer'], scale)}, from query_warehouse.")


def route(case: dict) -> str:
    return (f"Notes from earlier runs, retrieved for this question:\n"
            f"- 2025-10-02, asked \"{case['question']}\" and answered "
            "it with one query_warehouse call. No figure is kept: "
            "figures change.")


CAUTION = ("\nThese notes may be out of date. Use them to decide "
           "where to look, and take every figure from a tool.")
CONDITIONS = {
    "no note": lambda c: SYSTEM,
    "accurate note": lambda c: f"{SYSTEM}\n\n{note(c, 1.0)}",
    "stale note": lambda c: f"{SYSTEM}\n\n{note(c, DRIFT)}",
    "stale note, marked":
        lambda c: f"{SYSTEM}\n\n{note(c, DRIFT)}{CAUTION}",
    "route, no figure": lambda c: f"{SYSTEM}\n\n{route(c)}",
}
jobs = [(name, case) for name in CONDITIONS for case in cases for _ in range(RUNS)]
with ThreadPoolExecutor(max_workers=12) as pool:
    runs = list(pool.map(lambda job: agent.run(job[1]["question"], CONDITIONS[job[0]](job[1])),
                         jobs))

total = len(cases) * RUNS


def outcome(case: dict, run) -> tuple[bool, str]:
    """Did the warehouse return the figure, and what did the answer's first sentence state?"""
    returned = any(mentions(s.result, case["answer"]) for s in run.steps
                   if s.tool == "query_warehouse")
    first = re.split(r"(?<=[.!?])\s", run.answer.strip(), maxsplit=1)[0]
    if mentions(first, figure(case["answer"], DRIFT)):
        return returned, "note"
    if mentions(first, case["answer"]):
        return returned, "right"
    if any(len(re.sub(r"\D", "", n)) >= 5 for n in re.findall(r"\d[\d,.]*", first)):
        return returned, "other"
    return returned, "none"


def blended(case: dict, answer: str) -> bool:
    """The warehouse's digits with the note's first two in front of them."""
    true = re.sub(r"\D", "", case["answer"])
    note_digits = re.sub(r"\D", "", figure(case["answer"], DRIFT))
    return note_digits[:2] + true[2:] in re.sub(r"\D", "", answer)


KINDS = ("right", "note", "other", "none")
print(f"{len(cases)} warehouse questions with exact figures, {RUNS} runs each, "
      "v1.0's system prompt")
print(f"the stale note states a figure {DRIFT - 1:.0%} above the warehouse's\n")
print(f"  {'':<20}{'warehouse returned the figure':<30}{'warehouse returned null'}")
print((f"  {'memory':<20}" + f"{'right':>7}{'note':>6}{'other':>7}{'none':>6}    " * 2).rstrip())
summary, examples, answers = {}, {}, []
for name in CONDITIONS:
    rows = [(case, run) for (n, case), run in zip(jobs, runs) if n == name]
    counts = {(r, k): 0 for r in (True, False) for k in KINDS}
    for case, run in rows:
        returned, kind = outcome(case, run)
        counts[returned, kind] += 1
        if kind in ("note", "other"):
            examples.setdefault((returned, kind), (name, case, run))
        answers.append({"condition": name, "id": case["id"], "returned": returned,
                        "stated": kind, "answer": run.answer})
    line = ""
    for returned in (True, False):
        n = sum(counts[returned, k] for k in KINDS)
        note_cell = f"{counts[returned, 'note']}" if "stale" in name else "-"
        line += (f"{counts[returned, 'right']:>4}/{n:<2}{note_cell:>6}"
                 f"{counts[returned, 'other']:>7}{counts[returned, 'none']:>6}    ")
    print(f"  {name:<20}{line.rstrip()}")
    summary[name] = {f"{'returned' if r else 'null'}_{k}": v for (r, k), v in counts.items()}
    summary[name]["tokens"] = statistics.median(r.tokens for _, r in rows)

print("\n  the answer's first sentence stated the warehouse's figure (right), the note's")
print("  (note), some other figure (other), or no figure (none)")
wrong = [a for a in answers if a["stated"] in ("note", "other")]
cases_by_id = {c["id"]: c for c in cases}
blends = sum(blended(cases_by_id[a["id"]], a["answer"]) for a in wrong if a["stated"] == "other")
citing = sum("warehouse" in a["answer"].lower() for a in wrong)
print(f"\n  {len(wrong)} answers stated a figure the warehouse had not returned; "
      f"{citing} of them")
print("  still named the warehouse as the source")
others = sum(a["stated"] == "other" for a in wrong)
print(f"  {blends} of the {others} other figures were the warehouse's digits behind "
      "the note's")
print("  first two")
LABEL = {"note": "the note's figure", "other": "another figure"}
for (returned, kind), (name, case, run) in sorted(examples.items(), reverse=True):
    print(f"\n  {name}; warehouse returned {'the figure' if returned else 'null'}; "
          f"stated {LABEL[kind]}")
    print(f"    Q: {case['question']}")
    print(f"       warehouse {case['answer']}, note {figure(case['answer'], DRIFT)}")
    print("    A: " + "\n       ".join(textwrap.wrap(re.sub(r"\s+", " ", run.answer), 72)[:2]))

with open("code/17/_memory.json", "w") as f:
    json.dump({"summary": summary, "answers": answers}, f, indent=2)
