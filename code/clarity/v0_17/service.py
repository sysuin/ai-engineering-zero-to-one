"""
Clarity v0.17 — the engine, behind HTTP.

The engine does not change. What changes is everything around it, and the list is
longer than people expect: input that arrives from a stranger, a response that has to
start before the answer exists, a trace that has to survive the response, and a runtime
that is cooperative rather than preemptive.

One rule shapes the whole file. **The endpoint owns the request; the engine owns the
work.** Nothing in Clarity knows it is being served over HTTP, and nothing in here knows
how retrieval works. That boundary is what lets Chapter 19's MCP server, this API and
Chapter 32's command line be three front doors onto one system rather than three.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.platform.tracing import span, start                # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly.")


class AskRequest(BaseModel):
    """
    Validated at the edge, before anything expensive happens.

    A question of eight thousand words is not a question, it is either a mistake or an
    attack, and rejecting it here costs nothing. Chapter 26 adds the rest of the gates;
    this is the cheapest one and it belongs first.
    """
    question: str = Field(min_length=3, max_length=2000)
    max_steps: int = Field(default=6, ge=1, le=12)


class AskResponse(BaseModel):
    request_id: str
    answer: str
    tools_used: list[str]
    tokens: int
    seconds: float


def build_app(engine=None) -> FastAPI:
    app = FastAPI(title="Clarity", version="0.17.0")
    app.state.engine = engine

    def clarity():
        if app.state.engine is None:
            chunks, vectors = load_index()
            app.state.engine = build_tools(Retriever(chunks, vectors), Warehouse())
        return app.state.engine

    @app.get("/health")
    def health() -> dict:
        """Liveness only: is the process up. Readiness is Chapter 26's, and the
        difference between the two is the difference between a restart and a
        rebalance."""
        return {"status": "ok", "version": app.version}

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest,
                  request_id: Annotated[str | None, Header()] = None) -> AskResponse:
        rid = request_id or str(uuid.uuid4())
        started = time.perf_counter()
        # The engine is synchronous and does several seconds of waiting. Calling it
        # directly from an async endpoint blocks the event loop for every other
        # request in the process — §25.3 measures exactly how much.
        with span("clarity.request", **{"clarity.request_id": rid}):
            result = await asyncio.to_thread(
                lambda: Agent(clarity(), budget=Budget(steps=body.max_steps))
                .run(body.question, system=SYSTEM))
        return AskResponse(request_id=rid, answer=result.answer,
                           tools_used=[s.tool for s in result.steps if s.tool],
                           tokens=result.tokens,
                           seconds=round(time.perf_counter() - started, 2))

    @app.post("/ask/stream")
    async def ask_stream(body: AskRequest) -> StreamingResponse:
        """
        Server-sent events: one line per step, then the answer.

        The point is not speed — the total is identical. It is that a person sees the
        system working within a second instead of staring at a spinner for nine, and
        §28.7 is about how large that difference turns out to be.
        """
        rid = str(uuid.uuid4())

        async def events():
            yield _sse({"event": "accepted", "request_id": rid})
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()
            stopped = threading.Event()

            def put(item) -> None:
                loop.call_soon_threadsafe(queue.put_nowait, item)

            def work():
                # The span belongs to the work, not the transport: it opens before the
                # first step and closes after the answer, however the stream ends.
                with span("clarity.request", **{"clarity.request_id": rid,
                                                "clarity.streamed": True}):
                    try:
                        def report(step):
                            # As each step finishes, not after the run. An agent that
                            # reports progress at the end is buffering, not streaming.
                            if stopped.is_set():
                                raise _ClientGone()      # stop before the next model call
                            put({"event": "step", "tool": step.tool,
                                 "seconds": round(step.seconds, 2)})

                        agent = Agent(clarity(), budget=Budget(steps=body.max_steps))
                        run = agent.run(body.question, system=SYSTEM, on_step=report)
                        put({"event": "answer", "text": run.answer, "tokens": run.tokens})
                    except _ClientGone:
                        pass
                    except Exception as error:                    # noqa: BLE001
                        # Without this the stream would wait for an end that never comes.
                        put({"event": "error", "message": type(error).__name__})
                    finally:
                        put(None)

            # run_in_executor does not carry context variables across; copy them, or
            # the span above has no parent.
            loop.run_in_executor(None, contextvars.copy_context().run, work)
            try:
                while (item := await queue.get()) is not None:
                    yield _sse(item)
                yield _sse({"event": "done"})
            finally:
                # Reached when the stream ends normally and when the client disconnects
                # mid-stream. Either way, the agent stops at its next step instead of
                # spending tokens on an answer nobody will read.
                stopped.set()

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


class _ClientGone(Exception):
    """Raised inside the agent's step callback once nobody is listening."""


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"
