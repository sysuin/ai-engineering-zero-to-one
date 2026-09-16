# timeout: 1200
# The Chapter 17 agent, expressed as a graph. Same tools, same model, same question.

import sqlite3
import sys
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver

sys.path.insert(0, "code")
from clarity.v0_10.graph import build                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from meridian_index import load_index                    # noqa: E402

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())

STORE = Path("data/meridian/checkpoints.db")
STORE.unlink(missing_ok=True)
connection = sqlite3.connect(STORE, check_same_thread=False)
app = build(tools).compile(checkpointer=SqliteSaver(connection))

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool.")
QUESTION = "What was revenue in 2024 Q3, and what caused the Midwest decline?"

config = {"configurable": {"thread_id": "demo-1"}}
state = {"messages": [SystemMessage(content=SYSTEM),
                      HumanMessage(content=QUESTION)],
         "steps": 0, "budget": 8, "approvals": [], "stopped_because": ""}

print(f"Q: {QUESTION}\n")
for event in app.stream(state, config, stream_mode="updates"):
    for node, update in event.items():
        if node == "__interrupt__":
            continue
        messages = update.get("messages", []) if isinstance(update, dict) else []
        for m in messages:
            if getattr(m, "tool_calls", None):
                for call in m.tool_calls:
                    print(f"  think  -> {call['name']}({str(call['args'])[:44]})")
            elif m.__class__.__name__ == "ToolMessage":
                print(f"  act    <- {m.content[:64]}")
            elif m.content:
                print(f"  answer    {m.content[:180]}")

print(f"\ncheckpoints written: "
      f"{connection.execute('SELECT COUNT(*) FROM checkpoints').fetchone()[0]}")
print(f"store: {STORE}")          # its size is measured properly in 08_checkpoint_growth.py
print()
print("Every one of those rows is a resumable point. That is the whole difference, and")
print("it is not something you get by adding a feature to the loop — it comes from the")
print("framework owning the loop.")
