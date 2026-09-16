# Every checkpoint is a state you can return to. List a run's history, go back to a
# step, change what was decided there, and run forward again on a new branch.

import operator
import sqlite3
from typing import Annotated, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph


class State(TypedDict):
    quarter: str
    notes: Annotated[list[str], operator.add]


def plan(state: State) -> dict:
    return {"notes": [f"plan: brief {state['quarter']}"]}


def fetch(state: State) -> dict:
    return {"notes": [f"fetch: figures for {state['quarter']}"]}


def write(state: State) -> dict:
    return {"notes": [f"write: briefing for {state['quarter']}"]}


graph = StateGraph(State)
for name, node in (("plan", plan), ("fetch", fetch), ("write", write)):
    graph.add_node(name, node)
graph.add_edge(START, "plan")
graph.add_edge("plan", "fetch")
graph.add_edge("fetch", "write")
graph.add_edge("write", END)
connection = sqlite3.connect(":memory:", check_same_thread=False)
app = graph.compile(checkpointer=SqliteSaver(connection))

config = {"configurable": {"thread_id": "briefing"}}
app.invoke({"quarter": "2024 Q3", "notes": []}, config)

history = list(app.get_state_history(config))           # newest first
print(f"{len(history)} checkpoints in the run, oldest first:")
for snapshot in reversed(history):
    print(f"  next={str(snapshot.next):15} notes={len(snapshot.values.get('notes', []))}")

# Go back to the moment after `plan`, and change the decision that was made there.
after_plan = next(s for s in history if s.next == ("fetch",))
branch = app.update_state(after_plan.config, {"quarter": "2025 Q1"})
app.invoke(None, branch)

print("\nthe branch, as it now stands:")
for note in app.get_state(config).values["notes"]:
    print(f"  {note}")
print(f"\ncheckpoints now held for this thread: {len(list(app.get_state_history(config)))}")
print("the original run is still there:",
      any("write: briefing for 2024 Q3" in s.values.get("notes", [])
          for s in app.get_state_history(config)))
