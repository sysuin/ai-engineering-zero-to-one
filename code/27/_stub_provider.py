# skip
"""
A stand-in provider that speaks the other common wire shape.

It is not a model. It answers with a fixed script, and it exists so that Chapter 27's
adapter can be exercised end to end: the system prompt goes in a different place, the
content comes back as typed blocks, the usage fields have different names, and the stop
reason uses different words.

Testing an adapter against a stub is not testing a model. It is testing the translation,
which is the part that breaks, and the part you cannot test at all if the only way to
run your second adapter is to have a second account.
"""
from __future__ import annotations

import json


class BlockTransport:
    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.calls = 0
        self.last_payload: dict | None = None

    def messages(self, *, model, system, messages, tools, max_tokens,
                 temperature=None) -> dict:
        self.calls += 1
        self.last_payload = {"model": model, "system": system, "messages": messages,
                             "tools": tools, "max_tokens": max_tokens}
        if self.fail:
            raise ConnectionError("stub provider is unavailable")

        last = messages[-1]["content"] if messages else ""
        text = last if isinstance(last, str) else json.dumps(last)
        if tools and "revenue" in text.lower():
            return {"content": [{"type": "tool_use", "id": "toolu_01",
                                 "name": tools[0]["name"],
                                 "input": json.dumps({"question": text[:60]})}],
                    "usage": {"input_tokens": 41, "output_tokens": 17},
                    "stop_reason": "tool_use"}
        return {"content": [{"type": "text",
                             "text": "Answered by the block-shaped provider."}],
                "usage": {"input_tokens": 38, "output_tokens": 9},
                "stop_reason": "end_turn"}
