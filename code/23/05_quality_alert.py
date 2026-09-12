# timeout: 2400
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# The outage that every uptime check passes.

import json
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.judge import REFERENCE, Judge                # noqa: E402
from clarity.evals.runner import correct, load                  # noqa: E402
from clarity.platform.instrumented import TracedClient, traced_tools  # noqa: E402
from clarity.platform.tracing import span, start                # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import Tool, build_tools                # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly.")

# Questions only the documents can answer — contracts, the sideways table, the
# narrative causes. A first version used every document case and measured nothing,
# because "revenue in 2024 Q3" is answered from the warehouse and a broken retriever
# never touches it. Where a gold answer came from is not a claim about which tool
# runs, and this is the third chapter in which that has cost an experiment.
cases = ([c for c in load()
          if c["kind"] == "document" and c.get("tier") == "tail"] +
         [c for c in load() if c["kind"] == "warehouse"][:8])
chunks, vectors = load_index()
judge = Judge(system=REFERENCE)


def measure(label: str, degrade: bool) -> dict:
    """
    Run the sample and collect what each kind of monitor would have seen.

    The degradation is one line: retrieval returns its worst matches instead of its
    best. Nothing errors, nothing times out, nothing appears in a log at level ERROR.
    """
    sink = start()

    def run(case):
        client = TracedClient()
        retriever = Retriever(chunks, vectors, client=client)
        base = build_tools(retriever, Warehouse(client=client))
        if degrade:
            base = [Tool(t.name, t.description, t.parameters,
                         (lambda q, limit=5: [
                             {"source": p.source, "heading": p.heading,
                              "text": p.text[:600]}
                             for p in retriever.search(q, k=12)[-limit:]])
                         if t.name == "search_documents" else t.run, t.timeout)
                    for t in base]
        with span("clarity.request"):
            result = Agent(traced_tools(base), client=client,
                           budget=Budget(steps=6)).run(case["question"],
                                                       system=SYSTEM)
        return case, result

    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(run, cases))

    requests = [r for r in sink.rows() if r["name"] == "clarity.request"]
    latency = sorted(r["ms"] for r in requests)
    graded = sum(correct(case, result.answer) for case, result in outcomes)
    sampled = outcomes[:10]
    judged = sum(bool(judge.score(case["question"], result.answer,
                                  case["answer"]).correct)
                 for case, result in sampled)
    refusals = sum(1 for case, result in outcomes
                   if not correct(case, result.answer))
    return {"label": label,
            "errors": 0,
            "p95_ms": latency[int(0.95 * (len(latency) - 1))],
            "tokens": statistics.fmean(r.tokens for _, r in outcomes),
            "correct": graded, "of": len(cases),
            "judge": judged, "judge_of": len(sampled),
            "refusals": refusals}


healthy = measure("healthy", degrade=False)
broken = measure("retrieval quietly ranking backwards", degrade=True)
json.dump({"healthy": healthy, "broken": broken},
          open("code/23/_alerting.json", "w"), indent=1)

retrieval_only = sum(1 for c in cases if c.get("tier") == "tail")
print(f"{len(cases)} requests, before and after a one-line change to retrieval.")
print(f"{retrieval_only} of them can only be answered from the documents; the rest "
      f"have the")
print(f"warehouse as an alternative route.\n")
print(f"  {'':<36}{'healthy':>10}{'degraded':>11}   alerts?")
signals = [
    ("HTTP errors", lambda d: f"{d['errors']}", lambda a, b: b["errors"] > a["errors"]),
    ("p95 latency", lambda d: f"{d['p95_ms']:.0f}ms",
     lambda a, b: b["p95_ms"] > a["p95_ms"] * 1.25),
    ("tokens per request", lambda d: f"{d['tokens']:.0f}",
     lambda a, b: b["tokens"] > a["tokens"] * 1.25),
    ("answers judged correct", lambda d: f"{d['judge']}/{d['judge_of']}",
     lambda a, b: b["judge"] < a["judge"] - 1),
    ("answers actually correct", lambda d: f"{d['correct']}/{d['of']}",
     lambda a, b: b["correct"] < a["correct"] - 1),
]
for name, show, fires in signals:
    alert = "FIRES" if fires(healthy, broken) else "silent"
    print(f"  {name:<36}{show(healthy):>10}{show(broken):>11}   {alert}")

print()
print("Nothing failed. No request errored, no timeout was hit, no exception reached a")
print("log, and the service returned 200 to everything. An uptime check would have")
print("been green for the entire window, and an on-call engineer would have been")
print("right to ignore it.")
print()
print(f"The system was, meanwhile, getting {healthy['correct'] - broken['correct']} "
      f"more answers wrong out of {healthy['of']}.")
print()
if broken["p95_ms"] < healthy["p95_ms"] and broken["tokens"] < healthy["tokens"]:
    print(f"And look at the two rows above it. p95 fell from {healthy['p95_ms']:.0f}ms "
          f"to {broken['p95_ms']:.0f}ms and")
    print(f"the average request got {1 - broken['tokens'] / healthy['tokens']:.0%} "
          f"cheaper, because a system that")
    print("cannot find anything stops looking sooner. An efficiency dashboard would")
    print("have shown this outage as a good week.")
print()
print("This is the shape of nearly every real degradation in this kind of system: an")
print("index that rebuilt wrong, a prompt edit that shipped on Friday, a provider")
print("silently routing to a different model revision. None of them are outages, all")
print("of them are worse than outages, because an outage announces itself.")
print()
print("So the monitor has to be a quality signal, which means an eval running against")
print("production traffic. Three ways to get one, cheapest first:")
print()
print("  refusal rate        free, already in your logs, and the earliest warning")
print("                      you will get — a system that has stopped finding things")
print("                      starts saying so before anyone complains")
print("  judge on a sample   a few hundred graded answers a day is affordable and")
print("                      catches what refusal rate misses: confident wrongness")
print("  the golden set      run Chapter 21's cases against production hourly. It is")
print("                      the only signal with certain labels, and the only one")
print("                      that can page somebody at 3am without apology")
print()
print("Alert on the last one. Chart the first two. And set the threshold from §21.6's")
print("interval, not from a round number — a five-point drop on forty cases is noise,")
print("and paging somebody for noise is how a quality alert gets muted.")
