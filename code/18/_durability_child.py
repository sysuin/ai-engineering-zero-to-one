# skip
"""A worker for 11_durability.py: run a 200-step graph with one durability mode, timing it,
or dying without cleanup at step 100. An optional fourth argument makes each step take that many ms."""
import os
import sqlite3
import sys
import time
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

STORE, MODE, CRASH = sys.argv[1], sys.argv[2], sys.argv[3] == "crash"
PAUSE = float(sys.argv[4]) / 1000 if len(sys.argv) > 4 else 0.0
STEPS = 200


class State(TypedDict):
    step: int


def work(state):
    if CRASH and state["step"] == STEPS // 2:
        os._exit(9)
    time.sleep(PAUSE)                    # a node that does real work: a model call, a query
    return {"step": state["step"] + 1}


graph = StateGraph(State)
graph.add_node("work", work)
graph.add_edge(START, "work")
graph.add_conditional_edges("work", lambda s: END if s["step"] >= STEPS else "work")
app = graph.compile(checkpointer=SqliteSaver(sqlite3.connect(STORE, check_same_thread=False)))
config = {"configurable": {"thread_id": "long-run"}, "recursion_limit": 1000}

started = time.perf_counter()
app.invoke({"step": 0}, config, durability=MODE)
print(f"{(time.perf_counter() - started) / STEPS * 1000:.2f}")
