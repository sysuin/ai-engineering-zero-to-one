# Uses the cassette recorded by code/24/02_record_replay.py; re-run this after that one.
# "Replay production recordings against the next version" sounds like a regression suite that
# grows by itself. Seven small changes to Clarity, each replayed against one recorded run:
# how far does the replay get before the first request that was never recorded?

import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, "code")
from clarity.platform.replay import Cassette, RecordingClient  # noqa: E402
from clarity.v0_6.retrieve import Retriever                    # noqa: E402
from clarity.v0_7 import warehouse                             # noqa: E402
from clarity.v0_7.warehouse import Warehouse                   # noqa: E402
from clarity.v0_8.tools import build_tools                     # noqa: E402
from clarity.v0_9.agent import Agent, Budget                   # noqa: E402
from meridian_index import load_index                          # noqa: E402

TAPE = Path("data/meridian/cassettes/midwest.json")
QUESTION = "What was revenue in 2024 Q3, and what caused the Midwest decline?"
SYSTEM = "You are an analyst for Meridian. Use tools for every fact."
chunks, vectors = load_index()


def shorter_passages(tool):
    def run(**arguments):
        return [dict(p, text=p["text"][:500]) for p in tool.run(**arguments)]
    return dataclasses.replace(tool, run=run)


def reworded(tool):
    return dataclasses.replace(tool, description=tool.description.replace("passage", "excerpt"))


CHANGES = {
    "nothing":                        lambda tools: tools,
    "a longer timeout on every tool": lambda tools: [dataclasses.replace(t, timeout=30.0)
                                                     for t in tools],
    "a larger step budget":           "budget",
    "one word of a tool description": lambda tools: [reworded(t) if t.name == "search_documents"
                                                     else t for t in tools],
    "the tools listed in a new order": lambda tools: tools[::-1],
    "a trailing space in the prompt": "system",
    "passages cut at 500 characters": lambda tools: [shorter_passages(t)
                                                     if t.name == "search_documents"
                                                     else t for t in tools],
    "the revenue definition reworded": "definition",
}


class Watching(RecordingClient):
    """Notes which request missed first, and how many had been replayed before it."""
    first_miss: tuple[int, str] | None = None

    def _call(self, kind, fn, kwargs):
        try:
            return super()._call(kind, fn, kwargs)
        except KeyError:
            if self.first_miss is None:
                self.first_miss = (self.cassette.hits, kind)
            raise


recorded = len(Cassette(TAPE))
print(f"one recorded run of Clarity: {recorded} requests on the cassette\n")
print(f"  {'what changed':<33}{'replayed':>10}  first miss")
for name, change in CHANGES.items():
    cassette = Cassette(TAPE)
    client = Watching(cassette, mode="replay")
    tools = build_tools(Retriever(chunks, vectors, client=client), Warehouse(client=client))
    budget, system = Budget(steps=6), SYSTEM
    if change == "budget":
        budget = Budget(steps=10)
    elif change == "system":
        system = SYSTEM + " "
    elif change == "definition":          # the semantic layer is part of the warehouse's prompt
        note = warehouse.METRICS["revenue"]["note"]
        warehouse.METRICS["revenue"]["note"] = note.replace(". Excludes", "; excludes")
    else:
        tools = change(tools)
    try:
        Agent(tools, client=client, budget=budget).run(QUESTION, system=system)
    except KeyError:
        pass                              # a model call that was never recorded ends the run
    if change == "definition":
        warehouse.METRICS["revenue"]["note"] = note
    if client.first_miss is None:
        print(f"  {name:<33}{cassette.hits:>6} of {recorded}  none")
    else:
        hits, kind = client.first_miss
        request = "a model call" if kind != "embed" else "an embedding"
        print(f"  {name:<33}{hits:>6} of {recorded}  request {hits + 1}, {request}")
