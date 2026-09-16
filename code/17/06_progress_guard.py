# timeout: 2400
# A loop guard that compares arguments misses the loop that happens: the same search,
# reworded. This one watches results instead — a call that brings back nothing new
# is no progress, however it was phrased.

import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import Tool, ToolError, build_tools   # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

RUNS, LIMIT = 3, 2
SYSTEM = "You are an analyst for Meridian. Use tools for every fact."


def guarded(tool: Tool, limit: int = LIMIT) -> Tool:
    """The same tool, refusing to run after `limit` calls in a row found nothing new."""
    seen: set[tuple] = set()
    stale = {"calls": 0, "fired": False}

    def run(**arguments):
        if stale["calls"] >= limit:
            stale["fired"] = True
            raise ToolError(f"{limit} searches in a row found nothing new. Stop searching. "
                            "Answer with what you have, and say plainly what could not "
                            "be found.")
        result = tool.run(**arguments)
        found = {(p["source"], p["heading"]) for p in result}
        stale["calls"] = 0 if found - seen else stale["calls"] + 1
        seen.update(found)
        return result

    wrapped = replace(tool, run=run)
    wrapped.fired = lambda: stale["fired"]
    return wrapped


def never_finds(query: str, limit: int = 5) -> list:
    return []


EMPTY = Tool("search_documents", "Search Meridian's documents for a passage.",
             {"type": "object", "properties": {"query": {"type": "string"},
                                               "limit": {"type": "integer",
                                                         "default": 5}},
              "required": ["query"], "additionalProperties": False}, never_finds)
chunks, vectors = load_index()
retriever, warehouse = Retriever(chunks, vectors), Warehouse()

CASES = [
    ("a search that finds nothing", "Find Meridian's employee headcount.",
     lambda: [EMPTY]),
    ("the real corpus, no answer in it",
     "What is the average tenure of Meridian's warehouse staff? Be thorough.",
     lambda: build_tools(retriever, warehouse)),
]


def one(question, make_tools, guard: bool):
    toolset = make_tools()
    if guard:
        toolset = [guarded(t) if t.name == "search_documents" else t for t in toolset]
    run = Agent(toolset, budget=Budget(steps=8)).run(question, SYSTEM)
    fired = any(getattr(t, "fired", lambda: False)() for t in toolset)
    searches = run.tools_used.count("search_documents")
    refused = sum(s.failed for s in run.steps)
    return len(run.steps), searches, fired, run.stopped_because, refused


print(f"{RUNS} runs each; the guard allows {LIMIT} searches in a row that find nothing new\n")
print(f"  {'case':34} {'guard':>5} {'steps':>7} {'refused':>8} {'fired':>6}  stopped")
for label, question, make_tools in CASES:
    for guard in (False, True):
        with ThreadPoolExecutor(max_workers=RUNS) as pool:
            outcomes = list(pool.map(lambda _: one(question, make_tools, guard),
                                     range(RUNS)))
        steps = "/".join(str(o[0]) for o in outcomes)
        refused = "/".join(str(o[4]) for o in outcomes)
        fired = sum(o[2] for o in outcomes)
        stopped = sorted({o[3] for o in outcomes})
        print(f"  {label:34} {'on' if guard else 'off':>5} {steps:>7} {refused:>8} "
              f"{fired:>4}/{RUNS}  {', '.join(stopped)}")
