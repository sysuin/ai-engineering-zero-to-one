# skip
"""A run that deliberately dies halfway through. Started by 02_crash_and_resume.py."""
import os
import sqlite3
import sys

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, "code")
from clarity.v0_10.graph import build                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from meridian_index import load_index                    # noqa: E402

STORE, THREAD = sys.argv[1], sys.argv[2]
chunks, vectors = load_index()
connection = sqlite3.connect(STORE, check_same_thread=False)
app = build(build_tools(Retriever(chunks, vectors), Warehouse())).compile(
    checkpointer=SqliteSaver(connection))

state = {"messages": [
    SystemMessage(content="You are an analyst for Meridian. Use tools for every fact."),
    HumanMessage(content="Compare 2024 Q3 and 2025 Q1 on revenue and margin, and say "
                         "what the documents give as the cause of each movement.")],
    "steps": 0, "budget": 8, "approvals": [], "stopped_because": ""}

config = {"configurable": {"thread_id": THREAD}}
done = 0
for event in app.stream(state, config, stream_mode="updates"):
    for node in event:
        if node == "act":
            done += 1
            print(f"child: completed act #{done}", flush=True)
            if done == 1:
                print("child: dying now", flush=True)
                # Not an exception — the process is destroyed, as it would be by a
                # deploy, an OOM kill, or a machine going away.
                os._exit(9)
