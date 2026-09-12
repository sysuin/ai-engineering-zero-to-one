"""
Clarity v0.9 — the loop, with the four things that make it safe to run.

Chapter 16 built a tool loop. An agent is that loop plus:

    memory       what it has learned so far, carried between steps as data
    a budget     steps, seconds and money, all three enforced
    a stop rule  more than "the model stopped asking for tools"
    a trace      every decision recorded, because you cannot debug what you cannot read

Nothing here is a framework. It is about ninety lines, and Chapter 18 replaces it with
one so you can see what the framework was doing.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST                       # noqa: E402
from clarity.v0_8.tools import Tool, ToolError              # noqa: E402


@dataclass
class Step:
    n: int
    thought: str
    tool: str | None
    arguments: dict
    result: str
    seconds: float
    failed: bool = False


@dataclass
class Budget:
    """Three ceilings, because a run can exhaust any one of them first."""
    steps: int = 8
    seconds: float = 90.0
    tokens: int = 60_000

    def exceeded(self, steps: int, elapsed: float, tokens: int) -> str | None:
        if steps >= self.steps:
            return f"step budget ({self.steps}) exhausted"
        if elapsed >= self.seconds:
            return f"time budget ({self.seconds:.0f}s) exhausted"
        if tokens >= self.tokens:
            return f"token budget ({self.tokens:,}) exhausted"
        return None


@dataclass
class Run:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_because: str = "answered"
    tokens: int = 0
    seconds: float = 0.0

    @property
    def tools_used(self) -> list[str]:
        return [s.tool for s in self.steps if s.tool]


class Findings(BaseModel):
    """
    What the agent has established.

    Carried between steps as a typed object rather than as a growing transcript —
    Chapter 9 measured that difference at 72.5% against 95.0%. It is also what makes a
    run resumable, which Chapter 18 needs.
    """
    facts: list[str] = Field(default_factory=list,
                             description="Established facts, each with its source.")
    still_needed: list[str] = Field(default_factory=list,
                                    description="What remains unknown.")
    can_answer_now: bool = False


class Agent:
    def __init__(self, tools: list[Tool], client: OpenAI | None = None,
                 budget: Budget | None = None, model: str = MODEL_FAST):
        self.tools = {t.name: t for t in tools}
        self.client = client or OpenAI()
        self.budget = budget or Budget()
        self.model = model

    # ------------------------------------------------------------------ execution
    def _execute(self, name: str, arguments: dict) -> tuple[str, float, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f"No tool named {name!r}. Available: {', '.join(self.tools)}.", 0.0, True
        started = time.time()
        try:
            value = tool.run(**arguments)
            return json.dumps(value, default=str)[:3000], time.time() - started, False
        except ToolError as error:
            return str(error), time.time() - started, True
        except Exception as error:                                   # noqa: BLE001
            return f"{name} failed: {type(error).__name__}: {error}", \
                   time.time() - started, True

    # ------------------------------------------------------------------ the loop
    def run(self, question: str, system: str = "",
            on_step: Callable[[Step], None] | None = None) -> Run:
        """
        `on_step` is called as each step completes, not at the end.

        Added in Chapter 25 for streaming. An agent that reports its progress only
        after it has finished is not streaming, it is buffering — and the first
        version of Clarity's SSE endpoint did exactly that, emitting every step at
        once, five seconds in.
        """
        started = time.time()
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": question})

        run = Run(answer="")
        seen: set[tuple[str, str]] = set()

        while True:
            reason = self.budget.exceeded(len(run.steps), time.time() - started,
                                          run.tokens)
            if reason:
                run.stopped_because = reason
                run.answer = self._forced_answer(messages)
                break

            response = self.client.chat.completions.create(
                model=self.model, temperature=0, max_completion_tokens=700,
                messages=messages, tools=[t.schema() for t in self.tools.values()])
            run.tokens += response.usage.total_tokens
            reply = response.choices[0].message

            if not reply.tool_calls:
                run.answer = (reply.content or "").strip()
                break

            messages.append(reply)
            for call in reply.tool_calls:
                arguments = json.loads(call.function.arguments)
                signature = (call.function.name, json.dumps(arguments, sort_keys=True))

                if signature in seen:
                    # The same call with the same arguments cannot produce a different
                    # result. Saying so is cheaper than letting it discover that.
                    result, seconds, failed = (
                        f"You already called {call.function.name} with exactly these "
                        f"arguments and got a result above. Use it, change the "
                        f"arguments, or answer with what you have.", 0.0, True)
                else:
                    seen.add(signature)
                    result, seconds, failed = self._execute(call.function.name,
                                                            arguments)

                step = Step(len(run.steps) + 1, reply.content or "",
                            call.function.name, arguments,
                            result[:300], seconds, failed)
                run.steps.append(step)
                if on_step is not None:
                    on_step(step)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": result})

        run.seconds = time.time() - started
        return run

    def _forced_answer(self, messages: list[dict]) -> str:
        """When the budget runs out, say what is known rather than returning nothing."""
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, max_completion_tokens=400,
            messages=messages + [{"role": "user", "content":
                                  "You are out of budget. Answer with what you have "
                                  "established so far, and state plainly what is still "
                                  "missing. Do not call any more tools."}])
        return (response.choices[0].message.content or "").strip()
