# timeout: 900
# Stop mid-run, let the process end, and have a person answer later.

import json
import sqlite3
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

sys.path.insert(0, "code")
from clarity.v0_10.graph import build                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from meridian_index import load_index                    # noqa: E402

STORE = Path("data/meridian/approvals.db")
STORE.unlink(missing_ok=True)
connection = sqlite3.connect(STORE, check_same_thread=False)

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())

QUESTION = "What was revenue in 2024 Q3, and what do the documents say about it?"
START = {"messages": [
    SystemMessage(content="You are an analyst for Meridian. Use tools for every fact."),
    HumanMessage(content=QUESTION)],
    "steps": 0, "budget": 8, "approvals": [], "stopped_because": ""}


RECORD: dict[str, dict] = {}


def run(thread: str, answer: str) -> None:
    # The gate is a build-time decision, so the same agent can be strict in one
    # deployment and open in another without a code change.
    app = build(tools, needs_approval={"query_warehouse"}).compile(
        checkpointer=SqliteSaver(connection))
    config = {"configurable": {"thread_id": thread}}

    app.invoke(START, config)
    paused = app.get_state(config)
    request = paused.tasks[0].interrupts[0].value
    asked = ", ".join(f"{n}({a})" for n, a in zip(request["tools"],
                                                  request["arguments"]))
    print(f"  paused at  : {paused.next[0]}")
    print(f"  asking     : {asked}")

    # Here the process could exit for a week. Nothing is held in memory; the pending
    # question is a row in the store. This is a fresh graph object to prove it.
    later = build(tools, needs_approval={"query_warehouse"}).compile(
        checkpointer=SqliteSaver(connection))
    print(f"  person says: {answer!r}")
    final = later.invoke(Command(resume=answer), config)
    for line in final["approvals"]:
        print(f"  recorded   : {line}")
    print(f"  answer     : {final['messages'][-1].content.strip()[:220]}")
    RECORD[thread] = {"paused_at": paused.next[0], "asked": request["tools"],
                      "decision": answer, "approvals": final["approvals"],
                      "answer": final["messages"][-1].content.strip()}


print("=== a person approves ===")
run("approve-demo", "yes, go ahead")

print("\n=== a person refuses ===")
run("deny-demo", "no, the warehouse is mid-load and the numbers are wrong")

json.dump(RECORD, open("code/18/_approval.json", "w"), indent=1)

print()
print("Same graph, same question, two outcomes decided by someone who was not there")
print("when the run started. The mechanism works: the run stopped, a fresh graph object")
print("picked it up, and the denial came back as a tool result rather than an exception,")
print("so the model answered instead of crashing.")
print()
print("Now read the two answers again. The person refused the warehouse because the")
print("numbers were wrong, and the refused run reports the same revenue anyway - it")
print("came out of the quarterly review, which was never gated. The gate stopped a")
print("tool. It did not stop a fact. Chapter 26 is about the difference.")
