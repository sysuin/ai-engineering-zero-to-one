# skip
"""A worker for 10_two_resumes.py: pause a refund run, or resume it at an agreed instant."""
import sqlite3
import sys
import time
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

STORE, LEDGER, MODE, START_AT, LEASE = sys.argv[1:6]
NAME = sys.argv[6] if len(sys.argv) > 6 else "setup"
THREAD = "refund-order-O-55120"


class State(TypedDict):
    order: str
    decision: str


def approve(state):
    return {"decision": interrupt("refund 180.00?")}


def refund(state):
    with open(LEDGER, "a") as ledger:
        ledger.write(f"{state['order']} 180.00 by {NAME}\n")
    return {}


graph = StateGraph(State)
graph.add_node("approve", approve)
graph.add_node("refund", refund)
graph.add_edge(START, "approve")
graph.add_edge("approve", "refund")
graph.add_edge("refund", END)
connection = sqlite3.connect(STORE, check_same_thread=False, timeout=30)
app = graph.compile(checkpointer=SqliteSaver(connection))
config = {"configurable": {"thread_id": THREAD}}

if MODE == "pause":
    app.invoke({"order": "O-55120", "decision": ""}, config)
    sys.exit(0)

time.sleep(max(0.0, float(START_AT) - time.time()))
if LEASE == "lease":
    # The claim is a row with a primary key: only one INSERT can succeed.
    with sqlite3.connect(STORE, timeout=30) as db:
        db.execute("CREATE TABLE IF NOT EXISTS leases (thread TEXT PRIMARY KEY, holder TEXT)")
        try:
            db.execute("INSERT INTO leases VALUES (?, ?)", (THREAD, NAME))
        except sqlite3.IntegrityError:
            print(f"{NAME}: run already claimed, not resuming")
            sys.exit(0)
app.invoke(Command(resume="approved"), config)
print(f"{NAME}: resumed and finished")
