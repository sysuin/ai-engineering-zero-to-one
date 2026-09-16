"""
Clarity v0.19 — one interface, many providers.

Vendor lock-in is an architectural decision, not a procurement one. You do not choose
it in a meeting; you acquire it by letting a provider's response shape spread through
your codebase until `choices[0].message.tool_calls` appears in forty files.

The gateway is the one place that knows what a provider looks like. Above it,
everything speaks `Reply`. Below it, each adapter translates — and the translation is
larger than people expect, which is the point of §27.1.

What this interface deliberately does *not* expose: streaming granularity, provider
error classes, model-specific parameters. Every one of those is a place a provider
leaks through an abstraction that claimed to hide it.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str          # JSON text, because that is what every provider sends


@dataclass
class Reply:
    """
    What every provider's answer becomes.

    Four fields, and each one is normalised from somewhere different: the text, the
    tool calls, what it cost in tokens, and why it stopped. The last one is the field
    people forget, and it is the only way to tell a finished answer from a truncated
    one.
    """
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    tokens_in: int = 0
    tokens_out: int = 0
    stop_reason: str = "stop"        # stop | length | tool_calls | refusal
    provider: str = ""
    model: str = ""
    # Output tokens a reasoning model spent before its visible answer. They count against
    # the output ceiling, so a reply can stop at "length" with no text at all — §27.4.
    tokens_reasoning: int = 0


class Provider(Protocol):
    name: str

    def complete(self, messages: list[dict], tools: list[dict] | None = None,
                 max_tokens: int = 700, temperature: float | None = None) -> Reply:
        ...


class OpenAIProvider:
    """The adapter for the shape the rest of this book has been using."""

    name = "openai"

    def __init__(self, model: str, client=None) -> None:
        from openai import OpenAI
        self.model = model
        self._client = client or OpenAI()

    def complete(self, messages, tools=None, max_tokens=700,
                 temperature=None) -> Reply:
        extra = {}
        if temperature is not None:
            extra["temperature"] = temperature
        if tools:
            extra["tools"] = tools
        response = self._client.chat.completions.create(
            model=self.model, messages=messages,
            max_completion_tokens=max_tokens, **extra)
        choice = response.choices[0]
        message = choice.message
        return Reply(
            text=message.content or "",
            tool_calls=[ToolCall(c.id, c.function.name, c.function.arguments)
                        for c in (message.tool_calls or [])],
            tokens_in=response.usage.prompt_tokens,
            tokens_out=response.usage.completion_tokens,
            # OpenAI says "stop" / "length" / "tool_calls"; the names happen to match
            # ours, which will not be true of the next provider.
            stop_reason=choice.finish_reason or "stop",
            provider=self.name, model=self.model,
            tokens_reasoning=getattr(getattr(response.usage, "completion_tokens_details",
                                             None), "reasoning_tokens", None) or 0)


class BlockShapedProvider:
    """
    An adapter for the other common wire shape: system out of the message list,
    content as a list of typed blocks, usage under different names, and a stop reason
    with different words in it.

    This is the shape Anthropic's API uses and Bedrock's Converse API resembles. The
    adapter here is exercised in Chapter 27 against a local stand-in that speaks the
    same shape, so the *translation* is tested for real even where the model is not.
    """

    name = "block-shaped"

    STOP = {"end_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls",
            "refusal": "refusal"}

    def __init__(self, model: str, transport) -> None:
        self.model = model
        self._transport = transport      # anything with .messages(...) -> dict

    def complete(self, messages, tools=None, max_tokens=700,
                 temperature=None) -> Reply:
        # 1. The system prompt is a top-level argument here, not a message.
        system = "\n\n".join(m["content"] for m in messages
                             if m.get("role") == "system")
        body = [m for m in messages if m.get("role") != "system"]

        # 2. Tool results are content blocks on a user turn, not a `tool` role.
        converted = []
        for message in body:
            if message.get("role") == "tool":
                converted.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": message["tool_call_id"],
                     "content": message["content"]}]})
            elif message.get("role") == "assistant" and message.get("tool_calls"):
                blocks = []
                if message.get("content"):
                    blocks.append({"type": "text", "text": message["content"]})
                for call in message["tool_calls"]:
                    blocks.append({"type": "tool_use", "id": call["id"],
                                   "name": call["function"]["name"],
                                   "input": call["function"]["arguments"]})
                converted.append({"role": "assistant", "content": blocks})
            else:
                converted.append({"role": message["role"],
                                  "content": message.get("content") or ""})

        # 3. Tool schemas are flat here; OpenAI nests them under "function".
        flat = [{"name": t["function"]["name"],
                 "description": t["function"]["description"],
                 "input_schema": t["function"]["parameters"]} for t in (tools or [])]

        payload = self._transport.messages(model=self.model, system=system,
                                           messages=converted, tools=flat,
                                           max_tokens=max_tokens,
                                           temperature=temperature)

        # 4. And the reply comes back as blocks, with usage under other names.
        text = "".join(b.get("text", "") for b in payload["content"]
                       if b.get("type") == "text")
        calls = [ToolCall(b["id"], b["name"], b["input"])
                 for b in payload["content"] if b.get("type") == "tool_use"]
        usage = payload.get("usage", {})
        return Reply(text=text, tool_calls=calls,
                     tokens_in=usage.get("input_tokens", 0),
                     tokens_out=usage.get("output_tokens", 0),
                     stop_reason=self.STOP.get(payload.get("stop_reason", ""),
                                               "stop"),
                     provider=self.name, model=self.model)


class Gateway:
    """
    A primary, an ordered list of fallbacks, and a record of what it did.

    The fallback is not an error handler. It is a routing decision with a trace
    attached, because "the answer came from the other provider" is something the
    person reading the trace has to be able to see — §27.2.
    """

    def __init__(self, primary: Provider, fallbacks: list[Provider] | None = None,
                 guard_factory=None) -> None:
        self.primary = primary
        self.fallbacks = fallbacks or []
        # One guard per provider, not one for the gateway. A shared circuit breaker
        # is reset by whichever provider is healthy, so the dead primary is retried
        # on every request forever — which is exactly what the first version of this
        # class did, and the symptom was a breaker that never opened.
        self.guard_factory = guard_factory
        self.guards: dict[str, object] = {}
        self.attempts: list[tuple[str, str]] = []

    def _guard(self, name: str):
        if self.guard_factory is None:
            return None
        if name not in self.guards:
            self.guards[name] = self.guard_factory()
        return self.guards[name]

    def complete(self, messages, **kwargs) -> Reply:
        errors = []
        for provider in [self.primary, *self.fallbacks]:
            try:
                call = (lambda p=provider: p.complete(messages, **kwargs))
                guard = self._guard(provider.name)
                reply = guard(call) if guard else call()
                self.attempts.append((provider.name, "ok"))
                return reply
            except Exception as error:                       # noqa: BLE001
                self.attempts.append((provider.name, type(error).__name__))
                errors.append(f"{provider.name}: {error}")
        raise RuntimeError("every provider failed — " + "; ".join(errors))
