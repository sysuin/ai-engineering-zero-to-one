"""
Count every token a block of code is billed for, however deep inside it the call is made.

An agent reports the tokens of its own loop. It does not see the model calls its tools
make — the warehouse's query planner, the retriever's facet extractor, each embedding —
and a comparison between two designs that counts only what each reports is unfair to
whichever design makes fewer calls of its own. Chapter 17 found that the hard way. So
this patches the client classes themselves: every call made anywhere in the process,
on any thread, is added to the meter that is running.
"""
from __future__ import annotations

import threading
from contextlib import contextmanager

from openai.resources.chat.completions import Completions
from openai.resources.embeddings import Embeddings

_lock = threading.Lock()
_active: list["Meter"] = []


class Meter:
    def __init__(self) -> None:
        self.tokens = 0
        self.calls = 0


def _counted(original):
    def wrapper(self, *args, **kwargs):
        response = original(self, *args, **kwargs)
        usage = getattr(response, "usage", None)
        with _lock:
            for meter in _active:
                meter.calls += 1
                meter.tokens += getattr(usage, "total_tokens", 0) or 0
        return response
    return wrapper


if not getattr(Completions.create, "_metered", False):
    for cls, name in ((Completions, "create"), (Completions, "parse"),
                      (Embeddings, "create")):
        wrapped = _counted(getattr(cls, name))
        wrapped._metered = True
        setattr(cls, name, wrapped)


@contextmanager
def meter():
    """Everything billed inside the block. Meters must not overlap in time."""
    m = Meter()
    with _lock:
        _active.append(m)
    try:
        yield m
    finally:
        with _lock:
            _active.remove(m)
