# timeout: 300
# The drill from 06, rerun with the three changes the chapter names: a read timeout in seconds
# rather than minutes, fewer retries, and a 503 with Retry-After when the provider cannot be
# reached. Nothing in Clarity changes; the client it is given and the front door do.

import socket
import sys
import threading
import time

import httpx
from fastapi.responses import JSONResponse
from fastapi.testclient import TestClient
from openai import APIConnectionError, OpenAI

sys.path.insert(0, "code")
from clarity.v1_0.clarity import Clarity                         # noqa: E402
from clarity.v1_0.service import build_app                       # noqa: E402

QUESTION = "What was gross margin in 2023 Q1?"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def silent_server() -> int:
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(64)
    held = []

    def accept():
        while True:
            held.append(listener.accept()[0])

    threading.Thread(target=accept, daemon=True).start()
    return listener.getsockname()[1]


def engine_for(port: int) -> Clarity:
    client = OpenAI(
        base_url=f"http://127.0.0.1:{port}/v1", api_key="drill",
        timeout=httpx.Timeout(5.0, connect=2.0), max_retries=1)
    engine = Clarity(client=client)
    engine.engine()                      # load the index before timing
    return engine


def front_door(engine: Clarity) -> TestClient:
    app = build_app(engine)

    @app.exception_handler(APIConnectionError)   # timeouts included
    async def unavailable(request, error):
        return JSONResponse(
            {"detail": "the model provider is not answering"},
            status_code=503, headers={"Retry-After": "30"})
    return TestClient(app, raise_server_exceptions=False)


def timed(api: TestClient, question: str) -> str:
    started = time.perf_counter()
    response = api.post("/ask", json={"question": question})
    return (f"{response.status_code} ({response.json().get('detail')}) "
            f"after {time.perf_counter() - started:.1f}s")


print("1. The provider refuses connections\n")
print(f"  POST /ask -> {timed(front_door(engine_for(free_port())), QUESTION)}")

print("\n2. The provider accepts connections and never answers\n")
engine = engine_for(silent_server())
api = front_door(engine)
finished: list[float] = []
statuses: list[int] = []
t0 = time.perf_counter()


def hold(i: int) -> None:
    statuses.append(api.post("/ask", json={"question": f"{QUESTION} ({i})"}).status_code)
    finished.append(time.perf_counter() - t0)


waiting = [threading.Thread(target=hold, args=(i,), daemon=True)
           for i in range(engine.pool.limit)]
for thread in waiting:
    thread.start()
time.sleep(3)
print(f"  at 3s, with {engine.pool.limit} requests waiting, a new one: POST /ask -> "
      f"{timed(api, 'Anything?')}")
for thread in waiting:
    thread.join(timeout=120)
codes = ", ".join(f"{n} x {code}" for code, n in sorted(
    {c: statuses.count(c) for c in statuses}.items()))
print(f"  the {len(finished)} held requests ended between {min(finished):.0f}s and "
      f"{max(finished):.0f}s: {codes}")
print(f"  then a new request: POST /ask ->\n    {timed(api, 'Anything at all?')}")

print("\nBefore: a caller could wait 30 minutes, and a full bulkhead stayed full as long.")
print(f"After: nobody waits longer than about {max(finished):.0f}s, the slots come back on")
print("their own, and every caller is told the truth, with a Retry-After, instead of a")
print("500 or a hang.")
