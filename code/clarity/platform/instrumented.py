"""
Clarity v0.15 — the same system, wrapped in spans.

Nothing inside Clarity changed to produce the traces in Chapter 23. The agent, the
tools, the retriever and the warehouse are the files Parts III and IV built, and this
module wraps them from outside.

That is deliberate and it is §23.3's argument. Instrumentation that reaches into every
function is instrumentation somebody eventually rips out, because it has to be
maintained in step with the logic. Instrumentation at the boundaries — the model call,
the tool call, the request — survives refactors and covers the things that actually
cost time and money.
"""
from __future__ import annotations

import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.platform.tracing import record_model_call, span, tracer  # noqa: E402
from clarity.v0_8.tools import Tool                                   # noqa: E402


class _Proxy:
    """
    Delegate everything not explicitly wrapped.

    The first version of this file proxied `create` and nothing else. The warehouse
    calls `chat.completions.parse`, so passing the traced client into it raised
    AttributeError inside a tool, the tool returned an error string in three
    milliseconds, and the trace showed a warehouse that had become extremely fast.

    Instrumentation that changes behaviour is worse than none, because it is the
    thing you reach for when behaviour is already confusing. Delegate by default;
    wrap what you know about.
    """

    def __init__(self, parent: "TracedClient", inner) -> None:
        self.parent, self._inner = parent, inner

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Completions(_Proxy):
    def parse(self, **kwargs):
        parent = self.parent
        model = kwargs.get("model", "unknown")
        with span("model.parse") as current:
            response = self._inner.parse(**kwargs)
            parent.cost += record_model_call(current, model,
                                             getattr(response, "usage", None))
            parent.calls += 1
            current.set_attribute("clarity.schema",
                                  getattr(kwargs.get("response_format"),
                                          "__name__", "-"))
        return response

    def create(self, **kwargs):
        parent = self.parent
        model = kwargs.get("model", "unknown")
        with span("model.call") as current:
            response = self._inner.create(**kwargs)
            parent.cost += record_model_call(current, model,
                                             getattr(response, "usage", None))
            parent.calls += 1
            reply = response.choices[0].message
            # What the model decided, not what it was given. A span records the shape
            # of a step; §23.5 is about why it must not record the contents.
            current.set_attribute("clarity.tool_calls",
                                  ",".join(c.function.name
                                           for c in (reply.tool_calls or [])) or "-")
            current.set_attribute("clarity.answered", bool(reply.content))
        return response


class _Embeddings(_Proxy):
    def create(self, **kwargs):
        parent = self.parent
        with span("model.embed") as current:
            response = self._inner.create(**kwargs)
            parent.cost += record_model_call(current, kwargs.get("model", "unknown"),
                                             getattr(response, "usage", None))
            parent.embeds += 1
            current.set_attribute("clarity.inputs", len(kwargs.get("input", [])))
        return response


class _Chat(_Proxy):
    def __init__(self, parent: "TracedClient", inner) -> None:
        super().__init__(parent, inner)
        self.completions = _Completions(parent, inner.completions)


class TracedClient(_Proxy):
    """
    A stand-in for the OpenAI client that opens a span around every call it makes.

    Wrapping the client rather than the agent means every call is covered, including
    the ones made by code you did not write the instrumentation for — which §23.5
    shows is most of them.
    """

    def __init__(self, inner: OpenAI | None = None) -> None:
        self.parent = self
        self._inner = inner or OpenAI()
        self.chat = _Chat(self, self._inner.chat)
        self.embeddings = _Embeddings(self, self._inner.embeddings)
        self.cost = 0.0
        self.calls = 0
        self.embeds = 0


def traced_tools(tools: list[Tool]) -> list[Tool]:
    """Wrap each tool so the trace shows what ran, how long it took, and how much
    came back — never what came back."""
    def wrap(tool: Tool) -> Tool:
        def run(**arguments):
            with span(f"tool.{tool.name}") as current:
                for key, value in arguments.items():
                    current.set_attribute(f"clarity.arg.{key}", value)
                result = tool.run(**arguments)
                current.set_attribute("clarity.result_chars", len(str(result)))
                current.set_attribute("clarity.result_items",
                                      len(result) if isinstance(result, list) else 1)
                return result
        return Tool(tool.name, tool.description, tool.parameters, run, tool.timeout)
    return [wrap(t) for t in tools]
