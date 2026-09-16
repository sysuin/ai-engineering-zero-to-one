"""
Clarity v1.0 — the front door.

Chapter 25 built this endpoint against the engine directly. It now sits on top of
`Clarity`, which changed the file in exactly one way: the endpoint got shorter. That is
the test of a composition — if putting the pieces behind an object made the caller more
complicated, the object was the wrong shape.

Three routes, and each one is a different promise:

    GET  /health        the process is up. Nothing more. §26.10's distinction.
    POST /ask           a question, an answer, and everything needed to defend it
    POST /ask/stream    the same work, visible while it happens
"""
from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Annotated

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.platform.resilience import Open                      # noqa: E402
from clarity.platform.tracing import start                        # noqa: E402
from clarity.v1_0.clarity import Clarity                          # noqa: E402


class AskRequest(BaseModel):
    """Validated at the edge, before anything expensive happens."""
    question: str = Field(min_length=3, max_length=2000)
    max_steps: int = Field(default=6, ge=1, le=12)


class AskResponse(BaseModel):
    request_id: str
    trace_id: str
    answer: str
    tools_used: list[str]
    sources: list[str]
    tokens: int
    seconds: float
    cached: bool
    refused: bool


def build_app(engine: Clarity | None = None) -> FastAPI:
    app = FastAPI(title="Clarity", version="1.0.0")
    app.state.clarity = engine
    app.state.traces = start("clarity")

    def clarity() -> Clarity:
        if app.state.clarity is None:
            app.state.clarity = Clarity()
        return app.state.clarity

    @app.get("/health")
    def health() -> dict:
        """Liveness, not readiness. §26.10: the difference between the two is the
        difference between a restart and a rebalance."""
        return {"status": "ok", "version": app.version}

    @app.post("/ask", response_model=AskResponse)
    async def ask(body: AskRequest,
                  request_id: Annotated[str | None, Header()] = None) -> AskResponse:
        rid = request_id or str(uuid.uuid4())
        try:
            # The engine is synchronous and spends seconds waiting. Calling it from an
            # async endpoint without this blocks the event loop for every other request
            # in the process — §25.9 measures how much.
            answer = await asyncio.to_thread(clarity().ask, body.question)
        except Open:
            # The bulkhead is full. This is load shedding working, not an error: a 503
            # with a Retry-After is a truthful answer and a 30-second timeout is not.
            raise HTTPException(status_code=503, detail="busy",
                                headers={"Retry-After": "5"})
        return AskResponse(request_id=rid, trace_id=answer.trace_id,
                           answer=answer.text, tools_used=answer.tools,
                           sources=answer.sources, tokens=answer.tokens,
                           seconds=round(answer.seconds, 2), cached=answer.cached,
                           refused=answer.refused)

    @app.post("/ask/stream")
    async def ask_stream(body: AskRequest) -> StreamingResponse:
        """
        One event per step, as it completes.

        The total is identical to `/ask`. What changes is that a person sees the system
        working within a second instead of watching a spinner for nine — §25.4 measured
        that gap, and it is the cheapest latency improvement in the book.
        """
        rid = str(uuid.uuid4())

        async def events():
            yield _sse({"event": "accepted", "request_id": rid})
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def work():
                def report(step):
                    loop.call_soon_threadsafe(
                        queue.put_nowait,
                        {"event": "step", "n": step.n, "tool": step.tool,
                         "seconds": round(step.seconds, 2)})

                try:
                    answer = clarity().ask(body.question, on_step=report)
                    payload = {"event": "answer", "text": answer.text,
                               "sources": answer.sources, "tokens": answer.tokens,
                               "trace_id": answer.trace_id}
                except Open:
                    payload = {"event": "error", "detail": "busy"}
                loop.call_soon_threadsafe(queue.put_nowait, payload)
                loop.call_soon_threadsafe(queue.put_nowait, None)

            loop.run_in_executor(None, work)
            while (item := await queue.get()) is not None:
                yield _sse(item)
            yield _sse({"event": "done"})

        return StreamingResponse(events(), media_type="text/event-stream")

    return app


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


app = build_app()
