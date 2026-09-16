"""
Clarity v0.16 — record a run once, replay it for nothing.

A bug you cannot reproduce is a bug you cannot fix, and §24.1 measures why you often
cannot: the same input does not reliably give the same output. So stop trying to
reproduce the *run* and start reproducing the *recording*.

A cassette is every request this process made and the reply it got, keyed by the exact
request. Replaying it is deterministic by construction, costs nothing, and needs no
network — which means a failing run can go into a test suite, a colleague's laptop, and
a CI job without any of them paying for it again.

What it does not give you is a way to test a change to the prompt. A cassette answers
the question "what did the model say to *this*", and a changed prompt is a different
question. §24.6 uses it for exactly what it is good at: holding one stage fixed while
another varies.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def canonical(obj):
    """
    Reduce anything that might appear in a request to the parts that mean something.

    This exists because of a real failure. An agent appends the model's reply object
    to its own message list, so a recorded request carries the provider's message
    class and a replayed one carries our stand-in. Both say the same thing and
    neither serialises the same way, so the fourth request on the tape matched and
    the fifth did not.

    A cassette key has to be canonical rather than incidental: the role, the content,
    and the tool calls — not whatever fields the provider's SDK happens to attach
    this month.
    """
    if isinstance(obj, dict):
        return {k: canonical(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    # A class, not an instance. `response_format=QuerySpec` passes the type itself,
    # and every pydantic class has a `model_dump` attribute that explodes when called
    # unbound — which broke every tool while recording, not while replaying.
    if isinstance(obj, type):
        return obj.__name__
    if hasattr(obj, "role") and hasattr(obj, "content"):
        return {"role": obj.role, "content": obj.content,
                "tool_calls": [{"id": c.id, "name": c.function.name,
                                "arguments": c.function.arguments}
                               for c in (getattr(obj, "tool_calls", None) or [])]}
    if hasattr(obj, "model_dump"):
        return canonical(obj.model_dump())
    if hasattr(obj, "__name__"):
        return obj.__name__
    return str(obj)


def fingerprint(kwargs: dict) -> str:
    """
    The key. Every argument that could change the reply goes in, in a stable order.

    Leaving one out produces a cassette that silently returns the wrong recording,
    which is a worse debugging experience than no cassette at all.
    """
    payload = json.dumps(canonical(kwargs), sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class Cassette:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.entries: dict[str, dict] = {}
        if self.path.exists():
            self.entries = json.loads(self.path.read_text())
        self.hits = 0
        self.misses = 0

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=1))

    def __len__(self) -> int:
        return len(self.entries)


class _Completions:
    def __init__(self, parent: "RecordingClient", inner) -> None:
        self.parent, self._inner = parent, inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create(self, **kwargs):
        return self.parent._call("create", self._inner.create, kwargs)

    def parse(self, **kwargs):
        return self.parent._call("parse", self._inner.parse, kwargs)


class _Chat:
    def __init__(self, parent: "RecordingClient", inner) -> None:
        self.completions = _Completions(parent, inner.completions)


class _Embeddings:
    def __init__(self, parent: "RecordingClient", inner) -> None:
        self.parent, self._inner = parent, inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create(self, **kwargs):
        return self.parent._call("embed", self._inner.create, kwargs)


class RecordingClient:
    """
    A client that writes every exchange to a cassette, or reads from one.

    `mode="record"` calls the provider and stores the reply. `mode="replay"` never
    touches the network: a request that is not on the cassette raises rather than
    quietly falling through, because a replay that silently makes a live call is a
    replay that costs money and proves nothing.
    """

    def __init__(self, cassette: Cassette, mode: str = "record",
                 inner: OpenAI | None = None) -> None:
        self.cassette, self.mode = cassette, mode
        self._inner = inner or OpenAI()
        self.chat = _Chat(self, self._inner.chat)
        self.embeddings = _Embeddings(self, self._inner.embeddings)

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _call(self, kind: str, fn, kwargs: dict):
        key = fingerprint({"kind": kind, **kwargs})
        if self.mode == "replay":
            if key not in self.cassette.entries:
                self.cassette.misses += 1
                raise KeyError(
                    f"nothing recorded for this {kind} request. The cassette has "
                    f"{len(self.cassette)} entries and none of them match — which "
                    f"means the input changed, not the model.")
            self.cassette.hits += 1
            # The request names the schema class, so a replayed structured output can
            # be the real model rather than a dictionary that looks like one. An
            # earlier version returned a stand-in and every caller that touched a
            # nested field got 'dict object has no attribute'.
            return _rehydrate(kind, self.cassette.entries[key],
                              kwargs.get("response_format"))
        response = fn(**kwargs)
        self.cassette.entries[key] = _dehydrate(kind, response)
        return response


# --------------------------------------------------------------------- plumbing
class _Box:
    """A stand-in for the provider's response object, carrying only what we replay."""

    def __init__(self, data: dict) -> None:
        self.__dict__.update(data)


def _dehydrate(kind: str, response) -> dict:
    if kind == "embed":
        return {"kind": kind,
                "data": [{"embedding": d.embedding} for d in response.data],
                "usage": {"prompt_tokens":
                          getattr(response.usage, "prompt_tokens", 0)}}
    message = response.choices[0].message
    out = {"kind": kind, "content": message.content,
           "tool_calls": [{"id": c.id, "name": c.function.name,
                           "arguments": c.function.arguments}
                          for c in (message.tool_calls or [])],
           "usage": {"prompt_tokens": response.usage.prompt_tokens,
                     "completion_tokens": response.usage.completion_tokens,
                     "total_tokens": response.usage.total_tokens}}
    if kind == "parse":
        parsed = getattr(message, "parsed", None)
        out["parsed"] = parsed.model_dump() if parsed is not None else None
        out["parsed_type"] = type(parsed).__name__ if parsed is not None else None
    return out


def _rehydrate(kind: str, entry: dict, schema=None):
    usage = _Box(entry["usage"] | {"completion_tokens":
                                   entry["usage"].get("completion_tokens", 0),
                                   "total_tokens":
                                   entry["usage"].get("total_tokens", 0)})
    if kind == "embed":
        return _Box({"data": [_Box(d) for d in entry["data"]], "usage": usage})
    calls = [_Box({"id": c["id"], "type": "function",
                   "function": _Box({"name": c["name"],
                                     "arguments": c["arguments"]})})
             for c in entry["tool_calls"]]
    message = _Box({"content": entry["content"], "tool_calls": calls or None,
                    "role": "assistant", "parsed": entry.get("parsed")})
    if kind == "parse" and entry.get("parsed") is not None:
        message.parsed = (schema.model_validate(entry["parsed"])
                          if schema is not None else _Parsed(entry["parsed"]))
    return _Box({"choices": [_Box({"message": message, "finish_reason": "stop"})],
                 "usage": usage})


class _Parsed:
    """Replays a structured output as something with attribute access."""

    def __init__(self, data: dict) -> None:
        self._data = data
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self) -> dict:
        return dict(self._data)
