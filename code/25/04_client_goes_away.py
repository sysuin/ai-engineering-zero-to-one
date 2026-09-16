# timeout: 300
# A client that closes the stream halfway through. Does the server stop the work, or go on
# paying for an answer nobody will read? A stand-in agent keeps this free and exact.

import asyncio
import contextlib
import sys
import threading
import time

import httpx
import uvicorn

sys.path.insert(0, "code")
import clarity.v0_17.service as service                         # noqa: E402

STEPS, STEP_SECONDS, PORT = 8, 0.3, 8733
ran: list[int] = []


class StandInRun:
    def __init__(self, answer, tokens):
        self.answer, self.tokens = answer, tokens


class StandInAgent:
    """Eight steps of pretend work; every step would be a model call in the real agent."""

    def __init__(self, tools, budget=None):
        pass

    def run(self, question, system="", on_step=None):
        for n in range(1, STEPS + 1):
            time.sleep(STEP_SECONDS)
            ran.append(n)
            on_step(type("Step", (), {"tool": f"step {n}", "seconds": STEP_SECONDS}))
        return StandInRun("done", 0)


service.Agent = StandInAgent
app = service.build_app(engine=[])
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error"))
threading.Thread(target=server.run, daemon=True).start()
for _ in range(200):
    with contextlib.suppress(Exception):
        httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
        break
    time.sleep(0.05)


async def read_two_steps_and_leave() -> None:
    async with httpx.AsyncClient(timeout=30) as client:
        async with client.stream("POST", f"http://127.0.0.1:{PORT}/ask/stream",
                                 json={"question": "What was revenue in 2024 Q3?"}) as reply:
            seen = 0
            async for line in reply.aiter_lines():
                if '"step"' in line:
                    seen += 1
                    if seen == 2:
                        return                     # the user closed the tab


asyncio.run(read_two_steps_and_leave())
left_at = len(ran)
time.sleep(STEPS * STEP_SECONDS + 1)               # long enough for all eight to finish
print(f"the agent had {STEPS} steps of work; the client left after reading 2")
print(f"  steps finished when the client left:  {left_at}")
print(f"  steps finished in total:              {len(ran)}")
print(f"  steps that ran for nobody:            {len(ran) - left_at}")
