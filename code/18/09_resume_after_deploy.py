# A run paused on Monday resumes under Tuesday's code. What happens when the state's shape
# changed in between — and the migration that makes the resume mean what it meant.

import operator
import sqlite3
import tempfile
from pathlib import Path
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

STORE = Path(tempfile.mkdtemp()) / "runs.db"
CONFIG = {"configurable": {"thread_id": "briefing-7"}}


def build(state_type, fetch):
    def plan(state):
        return {"notes": ["plan: brief one quarter"]}

    def approve(state):
        return {"notes": [f"approved: {interrupt('fetch the figures?')}"]}

    graph = StateGraph(state_type)
    for name, node in (("plan", plan), ("approve", approve), ("fetch", fetch)):
        graph.add_node(name, node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "approve")
    graph.add_edge("approve", "fetch")
    graph.add_edge("fetch", END)
    return graph.compile(checkpointer=SqliteSaver(sqlite3.connect(STORE, check_same_thread=False)))


# ---- Monday's code: the quarter lives in `quarter`
class Monday(TypedDict):
    quarter: str
    notes: Annotated[list[str], operator.add]


monday = build(Monday, lambda s: {"notes": [f"fetched figures for {s['quarter']}"]})
monday.invoke({"quarter": "2024-Q3", "notes": []}, CONFIG)
print("Monday: paused at", monday.get_state(CONFIG).next, "with quarter =",
      monday.get_state(CONFIG).values.get("quarter"))


# ---- Tuesday's code: renamed to `period`
class Tuesday(TypedDict):
    period: str
    notes: Annotated[list[str], operator.add]


def attempt(label, app, command):
    try:
        result = app.invoke(command, CONFIG)
        print(f"  {label:38} finished: {result['notes'][-1]!r}")
    except Exception as error:                                   # noqa: BLE001
        print(f"  {label:38} {type(error).__name__}: {str(error)[:60]}")


print("\nTuesday resumes the same run:")
strict = build(Tuesday, lambda s: {"notes": [f"fetched figures for {s['period']}"]})
print("  what Tuesday's graph sees:", dict(strict.get_state(CONFIG).values))
attempt("fetch reads s['period']", strict, Command(resume="yes"))

forgiving = build(Tuesday, lambda s: {"notes": [f"fetched figures for {s.get('period', 'the latest quarter')}"]})
attempt("fetch reads s.get('period', default)", forgiving, Command(resume="yes"))

print("\nwith a migration, from the checkpoint where the run was paused:")
paused = next(h for h in strict.get_state_history(CONFIG) if h.next == ("approve",))
old = build(Monday, lambda s: {})
quarter = old.get_state(paused.config).values["quarter"]
migrated = build(Tuesday, lambda s: {"notes": [f"fetched figures for {s['period']}"]})
config = migrated.update_state(paused.config, {"period": quarter})
result = migrated.invoke(Command(resume="yes"), config)
print(f"  copied quarter={quarter!r} into period, then resumed: {result['notes'][-1]!r}")
