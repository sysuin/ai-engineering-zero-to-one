# timeout: 3000
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# Reads the golden set and runs Clarity five times over. The control's scores
# become the baseline every mutant is judged against.
# Prove the suite can fail before trusting it when it passes.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.runner import load                           # noqa: E402
from clarity.evals.suite import (DRIFT, green, run,             # noqa: E402
                                 scores)
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import Tool, build_tools                # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly rather than "
          "guessing.")
NO_ABSTAIN = "You are an analyst for Meridian. Use tools for every fact."

chunks, vectors = load_index()
retriever = Retriever(chunks, vectors)
warehouse = Warehouse()
base = build_tools(retriever, warehouse)

# A sample, balanced by hand: enough answerable cases to move layer 1, enough
# unanswerable ones that removing the refusal instruction shows up.
everything = load()
# All twenty refusals, not a sample of them. A floor over ten cases moves 10 points
# at a time, which is too coarse to distinguish a regression from a bad afternoon.
# Document cases the warehouse cannot answer — contracts, the sideways table, the
# narrative causes — so that a broken retriever has nowhere to hide.
cases = ([c for c in everything
          if c["kind"] == "document" and c.get("tier") == "tail"] +
         [c for c in everything
          if c["kind"] == "document" and c.get("tier") == "head"][:8] +
         [c for c in everything if c["kind"] == "warehouse"][:12] +
         [c for c in everything if c["kind"] == "unanswerable"])


def system_under_test(tools=None, system: str = SYSTEM):
    tools = tools or base

    def answer(question: str):
        # Record what every tool actually returned, so the retrieval layer can check
        # whether the passage it needed was ever in the room.
        seen: list[str] = []
        watched = [Tool(t.name, t.description, t.parameters,
                        (lambda run: lambda **kw: seen.append(
                            json.dumps(run(**kw), default=str)) or
                            json.loads(seen[-1]))(t.run),
                        t.timeout) for t in tools]
        result = Agent(watched, budget=Budget(steps=6)).run(question, system=system)
        return (result.answer, [s.tool for s in result.steps if s.tool],
                " ".join(seen))
    return answer


def wrap(name: str, fn):
    """Replace one tool's behaviour, leaving its name and schema untouched."""
    return [Tool(t.name, t.description, t.parameters,
                 fn if t.name == name else t.run, t.timeout) for t in base]


# Four plausible bugs and one control. Each is a single-line change of the kind that
# passes code review.
MUTANTS = {
    "control (nothing broken)": system_under_test(),
    "retrieval ranking reversed":
        system_under_test(wrap("search_documents",
                               lambda query, limit=5: [
                                   {"source": p.source, "heading": p.heading,
                                    "text": p.text[:600]}
                                   for p in reversed(retriever.search(query, k=10))
                               ][:limit])),
    "retrieval starved to one passage":
        system_under_test(wrap("search_documents",
                               lambda query, limit=5: [
                                   {"source": p.source, "heading": p.heading,
                                    "text": p.text[:600]}
                                   for p in retriever.search(query, k=1)])),
    "warehouse answers the previous quarter":
        system_under_test(wrap("query_warehouse", lambda question: (
            lambda shifted: {"rows": shifted.rows, "sql": shifted.sql,
                             "metric": shifted.spec.metric, "definition": shifted.note}
            if shifted else {"rows": [], "sql": "", "metric": "", "definition": ""}
        )(warehouse.ask(question.replace("Q4", "Q3").replace("Q3", "Q2")
                        .replace("Q2", "Q1"))))),
    "refusal instruction removed from the prompt":
        system_under_test(system=NO_ABSTAIN),
}

print(f"{len(cases)} cases, run against one healthy system and "
      f"{len(MUTANTS) - 1} broken ones.")
print(f"A layer fails if it drops more than {DRIFT:.0%} below the healthy run.\n")

# The control runs first and becomes the baseline. Comparing against a recorded
# number rather than a fixed floor is what makes this survive normal variance: every
# layer here is a sample, and a floor set just under today's score fails on Tuesdays.
control_layers = run(MUTANTS["control (nothing broken)"], cases=cases,
                     judge_sample=10)
baseline = scores(control_layers)

layer_names = ["deterministic", "grounded", "judge", "abstention", "retrieval",
               "warehouse"]
short = {"deterministic": "det", "grounded": "grnd", "judge": "judge",
         "abstention": "abst", "retrieval": "retr", "warehouse": "whse"}
header = "".join(f"{short[n]:>7}" for n in layer_names)
print(f"  {'':<28}{header}   build")
print(f"  {'baseline (the healthy run)':<28}"
      + "".join(f"{baseline[n]:>7.0%}" for n in layer_names) + "   —")

results = {"baseline": baseline}
for name, answerer in MUTANTS.items():
    if name.startswith("control"):
        continue
    layers = run(answerer, cases=cases, judge_sample=10, baseline=baseline)
    results[name] = {n: {"score": l.score, "ok": l.ok,
                         "drop": baseline[n] - l.score,
                         "failures": l.failures[:6]}
                     for n, l in layers.items()}
    print(f"  {name[:26]:<28}"
          + "".join(f"{layers[n].score:>7.0%}" for n in layer_names)
          + f"   {'green' if green(layers) else 'RED'}")

json.dump(results, open("code/22/_mutation.json", "w"), indent=1)

mutants = {n: r for n, r in results.items() if n != "baseline"}
survivors = [n for n, r in mutants.items() if all(l["ok"] for l in r.values())]
print()
print(f"  bugs the suite caught     {len(mutants) - len(survivors)} of {len(mutants)}")
print(f"  bugs it slept through     {len(survivors)}")
print()
for name, result in mutants.items():
    tripped = [n for n in layer_names if not result[n]["ok"]]
    worst = max(layer_names, key=lambda n: result[n]["drop"])
    if tripped:
        print(f"  {name}")
        print(f"    caught by {', '.join(tripped)} — {worst} fell "
              f"{result[worst]['drop']:.0%}")
if survivors:
    print()
    print("  survived:")
    for name in survivors:
        result = mutants[name]
        worst = max(layer_names, key=lambda n: result[n]["drop"])
        drop = result[worst]["drop"]
        print(f"    {name}")
        if drop <= 0.005:
            print(f"      nothing dropped anywhere — the largest fall on any layer "
                  f"was {max(drop, 0.0):.0%}")
            print(f"      which is the cheap case: either the mutation is not a bug, "
                  f"or it is one")
            print(f"      no layer here looks at. Read the mutation before deciding "
                  f"which.")
        else:
            print(f"      biggest drop: {worst} {drop:.0%}, which is "
                  f"{DRIFT - drop:.0%} short of the {DRIFT:.0%} band")
            print(f"      so the damage is real and diluted — confined to part of "
                  f"the set and")
            print(f"      divided by every case outside it. That needs a layer of "
                  f"its own,")
            print(f"      scoring {worst} on the population it affects, rather than "
                  f"a tighter band")
    print()
    print("  Those are the only two possibilities, and the size of the largest drop")
    print("  is what separates them. A tighter band would catch the second kind and")
    print("  make the build fail on ordinary variance; a new layer catches it and")
    print("  leaves the rest alone. Every layer in the table above arrived that way.")

# One mutant is worth a note whether or not it survived, because the result is the
# same either way and it is not the one people expect.
# One mutant is worth a note whether or not it survived, because which way it comes
# out is not stable between runs and both readings are instructive.
refusal = mutants.get("refusal instruction removed from the prompt")
if refusal:
    moved = baseline["abstention"] - refusal["abstention"]["score"]
    print()
    print(f"  Note the refusal row. Taking the sentence about refusing out of the")
    print(f"  prompt moved abstention by {moved:.0%}.")
    print()
    if moved < 0.05:
        print("  Which is to say: barely at all, on this run. The refusals were not only")
        print("  coming from that sentence: the warehouse raises an error naming what it")
        print("  does hold, and the passages that come back plainly do not answer — and")
        print("  the model reports both. One run cannot say whether the sentence never")
        print("  matters or only sometimes does, which is exactly why the slice is scored")
        print("  on its own and a surviving mutant is worth rerunning.")
    else:
        print("  So the sentence is load-bearing after all, and the tools alone do not")
        print("  carry the refusals. Run this again and the size of that number moves")
        print("  a good deal, which is its own lesson: a one-line prompt change can be")
        print("  worth fifteen points or nothing, and the only way to know which is a")
        print("  population scored on its own.")

print()
print("This is mutation testing, and it is the only evidence that a green suite means")
print("anything. A suite that has never been red is not passing — it is untested, and")
print("the difference is invisible until the day it matters.")
print()
print("Three versions of this suite were written before the table above, and each")
print("revision was dictated by a bug that walked past the one before it.")
print()
print("  three layers          caught one of four. The missing ones were a floor on")
print("                        refusals, which an overall average had absorbed, and")
print("                        one on retrieval, which a warehouse fallback hid.")
print("  absolute floors       failed its own control on the next run, because every")
print("                        layer is a sample and a floor an inch under today's")
print("                        score goes red on Tuesdays. Hence the baseline.")
print("  five layers           let the warehouse bug through: it moved the overall")
print("                        figure six points and sat inside the band, so the")
print("                        warehouse cases got a layer of their own.")
print()
print("Nobody designs the right suite first. You find out what it is missing by")
print("breaking things on purpose and watching what fails to notice.")
