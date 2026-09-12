"""
Clarity v0.10 — the same agent, as a graph.

Chapter 17's agent is ninety lines and works. This is the same behaviour expressed as a
state machine, and it exists for two things the ninety lines cannot do:

    a run survives the process dying halfway through
    a run can pause for a human who answers hours later

Both come from one mechanism — every step's state is written to a durable store — and
both are a great deal of work to build correctly by hand.

What it costs is a dependency with its own release cadence, and a stack trace that goes
through somebody else's code before it reaches yours.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated, Any, Literal, TypedDict

from langchain_core.messages import AIMessage, AnyMessage, HumanMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST                  # noqa: E402
from clarity.v0_8.tools import Tool, ToolError         # noqa: E402

# Which tools require a person to approve the call before it runs. Everything Clarity
# does today is read-only, so this is about the shape rather than the risk — Chapter 26
# adds the writes that make it matter.
DEFAULT_APPROVAL: set[str] = set()

# What the model is told when a person refuses. It has to be a result rather than an
# exception, or one declined call takes the whole run down with it.
DENIED = ("Denied by a person: {why}. Do not retry. Answer from what you already "
          "have, and say plainly what you could not check.")


def _pending(state: "State") -> list[dict]:
    """
    The tool calls from the latest request that nothing has answered yet.

    A denial answers some calls without running them, so `act` cannot simply take
    every call on the last message — it would run the ones a person just refused.
    """
    request = next(m for m in reversed(state["messages"])
                   if isinstance(m, AIMessage) and m.tool_calls)
    answered = {m.tool_call_id for m in state["messages"]
                if isinstance(m, ToolMessage)}
    return [c for c in request.tool_calls if c["id"] not in answered]


class State(TypedDict):
    """
    Everything the graph knows.

    `add_messages` is a reducer: nodes return the messages they produced and the
    framework appends them. Without it every node would have to receive and return the
    whole list, which is how hand-rolled versions acquire subtle ordering bugs.
    """
    messages: Annotated[list[AnyMessage], add_messages]
    steps: int
    budget: int
    approvals: list[str]
    stopped_because: str


def build(tools: list[Tool], client: OpenAI | None = None,
          model: str = MODEL_FAST, needs_approval: set[str] | None = None):
    client = client or OpenAI()
    needs_approval = DEFAULT_APPROVAL if needs_approval is None else needs_approval
    registry = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]

    # ---------------------------------------------------------------- nodes
    def think(state: State) -> dict:
        """Ask the model what to do next."""
        payload = []
        for m in state["messages"]:
            if isinstance(m, HumanMessage):
                payload.append({"role": "user", "content": m.content})
            elif isinstance(m, ToolMessage):
                payload.append({"role": "tool", "tool_call_id": m.tool_call_id,
                                "content": m.content})
            elif isinstance(m, AIMessage):
                entry: dict[str, Any] = {"role": "assistant",
                                         "content": m.content or None}
                if m.tool_calls:
                    entry["tool_calls"] = [
                        {"id": c["id"], "type": "function",
                         "function": {"name": c["name"],
                                      "arguments": json.dumps(c["args"])}}
                        for c in m.tool_calls]
                payload.append(entry)
            else:
                payload.append({"role": "system", "content": m.content})

        reply = client.chat.completions.create(
            model=model, temperature=0, max_completion_tokens=700,
            messages=payload, tools=schemas).choices[0].message

        calls = [{"name": c.function.name, "args": json.loads(c.function.arguments),
                  "id": c.id} for c in (reply.tool_calls or [])]
        return {"messages": [AIMessage(content=reply.content or "",
                                       tool_calls=calls)],
                "steps": state["steps"] + 1}

    def approve(state: State) -> dict:
        """
        Stop, and wait for a person.

        `interrupt` persists the state and hands control back
        to the caller, which may then exit. A resume continues
        from exactly here — not from the start of the run.
        """
        # Only gated calls go to a person. A model that asks for
        # the warehouse and a search in one turn should not have
        # the search blocked by a decision that was never about it.
        gated = [c for c in _pending(state) if c["name"] in needs_approval]
        decision = str(interrupt({"question": "Approve these calls?",
                                  "tools": [c["name"] for c in gated],
                                  "arguments": [c["args"] for c in gated]}))

        allowed = decision.strip().lower().startswith(("y", "approve", "ok"))
        verdict = "approved" if allowed else "denied"
        record = [f"{c['name']}: {verdict} ({decision})" for c in gated]
        if allowed:
            return {"approvals": record}

        # A refusal is a tool result, not an exception. The model sees why it was
        # stopped and gets to answer with what it already has.
        refusal = DENIED.format(why=decision)
        return {"approvals": record,
                "messages": [ToolMessage(tool_call_id=c["id"], content=refusal)
                             for c in gated]}

    def act(state: State) -> dict:
        """Run the requested tools and return their results."""
        results = []
        for call in _pending(state):
            tool = registry.get(call["name"])
            if tool is None:
                content = f"No tool named {call['name']!r}."
            else:
                try:
                    content = json.dumps(tool.run(**call["args"]), default=str)[:3000]
                except ToolError as error:
                    content = str(error)
                except Exception as error:                       # noqa: BLE001
                    content = f"{call['name']} failed: {type(error).__name__}: {error}"
            results.append(ToolMessage(content=content, tool_call_id=call["id"]))
        return {"messages": results}

    def give_up(state: State) -> dict:
        return {"stopped_because": f"step budget ({state['budget']}) exhausted"}

    # ---------------------------------------------------------------- edges
    def next_step(state: State) -> Literal["approve", "act", "give_up", "__end__"]:
        last = state["messages"][-1]
        if not getattr(last, "tool_calls", None):
            return END
        if state["steps"] >= state["budget"]:
            return "give_up"
        if any(c["name"] in needs_approval for c in last.tool_calls):
            return "approve"
        return "act"

    def after_approval(state: State) -> Literal["act", "think"]:
        """Run whatever the denial did not already answer; if nothing is left, think."""
        return "act" if _pending(state) else "think"

    graph = StateGraph(State)
    graph.add_node("think", think)
    graph.add_node("approve", approve)
    graph.add_node("act", act)
    graph.add_node("give_up", give_up)

    graph.add_edge(START, "think")
    graph.add_conditional_edges("think", next_step)
    graph.add_conditional_edges("approve", after_approval)
    graph.add_edge("act", "think")          # the loop, as one line
    graph.add_edge("give_up", END)
    return graph
