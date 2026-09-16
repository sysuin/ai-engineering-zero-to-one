# What a checkpoint store holds. A run whose state grows by one tool result per step,
# written to SQLite: how many rows, how many bytes, and how the bytes grow with steps.

import sqlite3
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

RESULT = "x" * 2_000                    # about the size of one search result


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    limit: int


def step(state: State) -> dict:
    n = state["steps"] + 1
    call = {"name": "search_documents", "args": {"query": f"q{n}"}, "id": f"call-{n}"}
    return {"messages": [AIMessage(content="", tool_calls=[call]),
                         ToolMessage(content=RESULT, tool_call_id=f"call-{n}")],
            "steps": n}


graph = StateGraph(State)
graph.add_node("step", step)
graph.add_edge(START, "step")
graph.add_conditional_edges("step", lambda s: END if s["steps"] >= s["limit"] else "step")


def stored_bytes(connection: sqlite3.Connection) -> int:
    """The serialised state the store keeps: checkpoints plus pending writes."""
    (checkpoints,) = connection.execute(
        "SELECT SUM(LENGTH(checkpoint) + LENGTH(metadata)) FROM checkpoints").fetchone()
    (writes,) = connection.execute("SELECT SUM(LENGTH(value)) FROM writes").fetchone()
    return (checkpoints or 0) + (writes or 0)


print(f"state grows by one {len(RESULT):,}-character result per step\n")
print(f"  {'steps':>5} {'rows':>6} {'stored':>10} {'per step':>10}")
previous = None
for limit in (5, 10, 20, 40):
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    app = graph.compile(checkpointer=SqliteSaver(connection))
    config = {"configurable": {"thread_id": "t"}, "recursion_limit": 1000}
    app.invoke({"messages": [], "steps": 0, "limit": limit}, config)
    rows = connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    size = stored_bytes(connection)
    print(f"  {limit:>5} {rows:>6} {size / 1024:>7,.0f} KB {size / limit / 1024:>7,.1f} KB")
    previous = (limit, size) if previous is None else previous
first_limit, first_size = previous
growth = (size / first_size) / (limit / first_limit)
print(f"\n{limit / first_limit:.0f}x the steps stored {size / first_size:.0f}x the bytes: "
      f"{'faster than' if growth > 1.5 else 'in line with'} the number of steps")
