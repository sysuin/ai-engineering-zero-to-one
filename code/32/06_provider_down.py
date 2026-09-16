# timeout: 300
# An incident drill: Clarity v1.0 with its provider refusing connections, then with its
# provider accepting connections and never answering. No tokens are spent — nothing on
# the other end is a model.

import socket
import sys
import threading
import time

from fastapi.testclient import TestClient
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.platform.resilience import Open                     # noqa: E402
from clarity.v1_0.clarity import Clarity                         # noqa: E402
from clarity.v1_0.service import build_app                       # noqa: E402

QUESTION = "What was gross margin in 2023 Q1?"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def silent_server() -> int:
    """Accepts every connection and never sends a byte: a provider that is slow, not down."""
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(64)
    held = []

    def accept():
        while True:
            held.append(listener.accept()[0])

    threading.Thread(target=accept, daemon=True).start()
    return listener.getsockname()[1]


def pointed_at(port: int) -> Clarity:
    client = OpenAI(base_url=f"http://127.0.0.1:{port}/v1", api_key="drill")
    return Clarity(client=client)


# ------------------------------------------------------------------ 1. down
print("1. The provider refuses connections\n")
engine = pointed_at(free_port())
engine.engine()                                  # load the index before timing anything
api = TestClient(build_app(engine), raise_server_exceptions=False)
started = time.perf_counter()
response = api.post("/ask", json={"question": QUESTION})
down_seconds = time.perf_counter() - started
print(f"  POST /ask -> {response.status_code} after {down_seconds:.1f}s")
try:
    engine.ask(QUESTION)
except Exception as error:                        # noqa: BLE001 — the drill is the point
    print(f"  underneath: {type(error).__name__}, after the SDK's "
          f"{engine.client._inner.max_retries} retries")

engine.cache.put(QUESTION, "(an answer cached before the outage)", tenant=engine.tenant)
again = engine.ask(QUESTION)
print(f"  the same question, cached earlier: answered={not again.refused}, "
      f"cached={again.cached}")

# ------------------------------------------------------------------ 2. slow
print("\n2. The provider accepts connections and never answers\n")
engine = pointed_at(silent_server())
engine.engine()
waiting = [threading.Thread(target=engine.ask, args=(f"{QUESTION} ({i})",), daemon=True)
           for i in range(engine.pool.limit)]
for thread in waiting:
    thread.start()
time.sleep(10)
alive = sum(thread.is_alive() for thread in waiting)
timeout = engine.client._inner.timeout.read
print(f"  after 10s, {alive} of {len(waiting)} requests are still waiting; each attempt")
print(f"  may wait {timeout:.0f}s, and the SDK makes "
      f"{engine.client._inner.max_retries + 1} attempts")

api = TestClient(build_app(engine), raise_server_exceptions=False)
started = time.perf_counter()
response = api.post("/ask", json={"question": "Anything at all?"})
print(f"  a new request: POST /ask -> {response.status_code} "
      f"({response.json().get('detail')}) after {time.perf_counter() - started:.2f}s")
print(f"  the bulkhead has turned away {engine.pool.rejected} so far")

worst = timeout * (engine.client._inner.max_retries + 1) / 60
print(f"\nDown is loud: a 500 in {down_seconds:.1f}s. Slow is quiet: every slot held for up "
      f"to {worst:.0f} minutes,")
print("and everyone else refused until then. The runbook needs both.")
