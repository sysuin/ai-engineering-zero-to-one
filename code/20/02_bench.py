# timeout: 3600
# One agent against three, on eight questions that need both halves of Meridian.

import json
import sys
import time
from statistics import median

sys.path.insert(0, "code")
sys.path.insert(0, "code/20")
from _meter import meter                                 # noqa: E402
from _taskset import TASKS, score                        # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from clarity.v0_12.team import Team                      # noqa: E402
from meridian_index import load_index                    # noqa: E402

REPEATS = 3
SOLO = ("You are an analyst for Meridian. Use tools for every fact. Give the figure "
        "first, then the cause, then anything you could not verify.")

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())
team = Team(tools)


# Tokens and calls are counted at the client, so the model calls inside the tools — the
# planner, the facet extractor, the embeddings — count for both designs.
def run_solo(question: str):
    started = time.perf_counter()
    with meter() as m:
        run = Agent(tools, budget=Budget(steps=8)).run(question, system=SOLO)
    return run.answer, m.tokens, time.perf_counter() - started, m.calls


def run_team(question: str):
    with meter() as m:
        run = team.run(question)
    return run.answer, m.tokens, run.seconds, m.calls


results = {}
for label, runner in (("one agent", run_solo), ("supervisor + 2", run_team)):
    figures = causes = 0
    tokens, seconds, calls = [], [], []
    per_task, missed = [], []
    for task in TASKS:
        got_figure = got_cause = 0
        for _ in range(REPEATS):
            answer, used, took, made = runner(task["q"])
            f, c = score(task, answer)
            if not (f and c):
                missed.append({"q": task["q"], "figure": f, "cause": c, "answer": answer})
            got_figure += f
            got_cause += c
            tokens.append(used)
            seconds.append(took)
            calls.append(made)
        figures += got_figure
        causes += got_cause
        per_task.append({"q": task["q"][:52], "figure": got_figure,
                         "cause": got_cause})
    total = len(TASKS) * REPEATS
    results[label] = {"figure": figures, "cause": causes, "of": total, "missed": missed,
                      "tokens": round(sum(tokens) / len(tokens)),
                      "seconds": round(median(seconds), 1),
                      "calls": round(sum(calls) / len(calls), 1),
                      "per_task": per_task}
    row = results[label]
    print(f"{label:<14} fig {row['figure']:>2}/{total}  "
          f"cause {row['cause']:>2}/{total}  "
          f"{row['tokens']:>6,} tok  {row['seconds']:>5.1f}s  "
          f"{row['calls']:>4.1f} calls")

json.dump(results, open("code/20/_bench.json", "w"), indent=1)

solo, multi = results["one agent"], results["supervisor + 2"]
correct = {k: (v["figure"] + v["cause"]) / (2 * v["of"]) for k, v in results.items()}
print()
print(f"accuracy   one agent {correct['one agent']:.0%}   "
      f"team {correct['supervisor + 2']:.0%}")
print(f"tokens     {multi['tokens'] / solo['tokens']:.1f}x")
print(f"latency    {multi['seconds'] / solo['seconds']:.1f}x")
print(f"calls      {multi['calls'] / solo['calls']:.1f}x")
print()
print("Per task, where they differ:")
for a, b in zip(solo["per_task"], multi["per_task"]):
    if (a["figure"], a["cause"]) != (b["figure"], b["cause"]):
        print(f"  {a['q']}…")
        print(f"     one agent  figure {a['figure']}/{REPEATS}  "
              f"cause {a['cause']}/{REPEATS}")
        print(f"     team       figure {b['figure']}/{REPEATS}  "
              f"cause {b['cause']}/{REPEATS}")

lost = [m for m in multi["missed"] if not m["figure"]]
if lost:
    print("\nWhere the team lost a figure, in the first line of its own answer:")
    for m in lost:
        first = next((line for line in m["answer"].splitlines() if line.strip()), "")
        print(f"  {m['q'][:44]}…")
        print(f"     {first.strip()[:74]}")
