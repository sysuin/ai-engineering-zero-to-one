"""
Clarity v0.9 — the loop, with the four things that make it safe to run.

Chapter 16 built a tool loop. An agent is that loop plus:

    memory       the conversation so far — the simplest kind; §17.8 measures it
    a budget     steps, seconds and tokens, all three enforced
    a stop rule  more than "the model stopped asking for tools"
    a trace      every decision recorded, because you cannot debug what you cannot read

Nothing here is a framework. The loop is about ninety lines, and Chapter 18 ports it to
one so you can see what the framework is doing — and v1.0 still runs on this one.
"""
from __future__ import annotations

import contextvars
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST                       # noqa: E402
from clarity.v0_8.tools import Tool, ToolError, ToolRunner  # noqa: E402


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
    """
    Three ceilings, because a run can exhaust any one of them first.

    They are checked between rounds, so a reply that asks for several tools at once
    can finish that reply a few steps past the step ceiling.
    """
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


class Agent:
    def __init__(self, tools: list[Tool], client: OpenAI | None = None,
                 budget: Budget | None = None, model: str = MODEL_FAST):
        self.tools = {t.name: t for t in tools}
        self.client = client or OpenAI()
        self.budget = budget or Budget()
        self.model = model

    # ------------------------------------------------------------------ execution
    def _execute(self, name: str, arguments: dict | None) -> tuple[str, float, bool]:
        tool = self.tools.get(name)
        if tool is None:
            return f"No tool named {name!r}. Available: {', '.join(self.tools)}.", 0.0, True
        if arguments is None:
            return (f"The arguments for {name} were not a JSON object. Retry with "
                    "arguments that match the schema.", 0.0, True)
        started = time.time()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            # The tool's own timeout, enforced without waiting for a hung thread
            # (§16.12 shows why a with-block would wait).
            # A thread does not inherit context variables, so the tracing span in
            # progress would not be the tool's parent; copy the context across.
            context = contextvars.copy_context()
            value = pool.submit(context.run, tool.run, **arguments).result(timeout=tool.timeout)
            return json.dumps(value, default=str)[:3000], time.time() - started, False
        except FutureTimeout:
            return (f"{name} took longer than {tool.timeout}s and was abandoned. "
                    "Try a narrower request.", time.time() - started, True)
        except ToolError as error:
            return str(error), time.time() - started, True
        except Exception as error:                                   # noqa: BLE001
            return f"{name} failed: {type(error).__name__}: {error}", \
                   time.time() - started, True
        finally:
            pool.shutdown(wait=False)

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
                run.answer, spent = self._forced_answer(messages)
                run.tokens += spent
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
                arguments = ToolRunner._parse(call.function.arguments)
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
                            call.function.name, arguments or {},
                            result[:300], seconds, failed)
                run.steps.append(step)
                if on_step is not None:
                    on_step(step)
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": result})

        run.seconds = time.time() - started
        return run

    def _forced_answer(self, messages: list[dict]) -> tuple[str, int]:
        """When the budget runs out, say what is known rather than returning nothing."""
        response = self.client.chat.completions.create(
            model=self.model, temperature=0, max_completion_tokens=400,
            messages=messages + [{"role": "user", "content":
                                  "You are out of budget. Answer with what you have "
                                  "established so far, and state plainly what is still "
                                  "missing. Do not call any more tools."}])
        # The forced answer is a model call like any other, and its tokens count.
        return ((response.choices[0].message.content or "").strip(),
                response.usage.total_tokens)
