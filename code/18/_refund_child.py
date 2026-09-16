# skip
"""A refund that dies halfway through. Started by 06_replayed_side_effects.py."""
import json
import os
import sqlite3
import sys
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

STORE, LEDGER, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
CRASH = sys.argv[4] == "crash"


class State(TypedDict):
    order: str
    amount: float
    refunded: bool


def already_done(key: str) -> bool:
    if not os.path.exists(LEDGER):
        return False
    return any(json.loads(line)["key"] == key for line in open(LEDGER))


def refund(state: State) -> dict:
    key = f"refund:{state['order']}"        # the same on every attempt at this refund
    if MODE == "idempotent" and already_done(key):
        return {"refunded": True}           # done before the crash: do not repeat it
    with open(LEDGER, "a") as ledger:       # the side effect: money leaves
        ledger.write(json.dumps({"key": key, "amount": state["amount"]}) + "\n")
    if CRASH:
        os._exit(9)                         # dies before the node can return
    return {"refunded": True}


def notify(state: State) -> dict:
    return {}


graph = StateGraph(State)
graph.add_node("refund", refund)
graph.add_node("notify", notify)
graph.add_edge(START, "refund")
graph.add_edge("refund", "notify")
graph.add_edge("notify", END)

connection = sqlite3.connect(STORE, check_same_thread=False)
app = graph.compile(checkpointer=SqliteSaver(connection))
config = {"configurable": {"thread_id": "order-1042"}}
start = {"order": "ORD-1042", "amount": 180.0, "refunded": False}
resuming = bool(app.get_state(config).next)
for _ in app.stream(None if resuming else start, config):
    pass
print(f"finished: refunded={app.get_state(config).values['refunded']}")
