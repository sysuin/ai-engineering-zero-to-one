# A client whose request timed out does not know whether the work happened, so it sends the
# request again. For an answer that is waste; for anything that writes, it is a duplicate.
# An idempotency key makes the retry safe. A stand-in for the expensive work keeps it free.

import threading
import time

from fastapi import FastAPI, Header, HTTPException
from fastapi.testclient import TestClient

work_done: list[str] = []


def expensive(question: str) -> str:
    time.sleep(0.2)                          # the model calls, or a write somebody pays for
    work_done.append(question)
    return f"answer to {question!r}"


def build(with_keys: bool) -> FastAPI:
    app = FastAPI()
    results: dict[str, dict] = {}            # in production: a shared store with a TTL
    in_progress: set[str] = set()
    lock = threading.Lock()

    @app.post("/ask")
    def ask(body: dict, idempotency_key: str | None = Header(default=None)) -> dict:
        if not with_keys or idempotency_key is None:
            return {"answer": expensive(body["question"])}
        with lock:
            if idempotency_key in results:
                return results[idempotency_key] | {"replayed": True}
            if idempotency_key in in_progress:
                # Still running: say so, rather than starting a second copy.
                raise HTTPException(409, "a request with this key is still in progress",
                                    headers={"Retry-After": "1"})
            in_progress.add(idempotency_key)
        try:
            result = {"answer": expensive(body["question"])}
            with lock:
                results[idempotency_key] = result
            return result
        finally:
            with lock:
                in_progress.discard(idempotency_key)

    return app


def client_with_retries(app: FastAPI, key: str | None) -> list[int]:
    """The first attempt 'times out' on the client side while the server keeps working."""
    client = TestClient(app)
    headers = {"Idempotency-Key": key} if key else {}
    statuses = []
    first = threading.Thread(target=lambda: statuses.append(
        client.post("/ask", json={"question": "refund order 1042"}, headers=headers)
        .status_code))
    first.start()
    time.sleep(0.05)                          # the client gives up waiting and retries
    statuses.append(client.post("/ask", json={"question": "refund order 1042"},
                                headers=headers).status_code)
    first.join()
    time.sleep(0.3)                           # and retries once more, later
    statuses.append(client.post("/ask", json={"question": "refund order 1042"},
                                headers=headers).status_code)
    return statuses


for label, app, key in (("no key", build(False), None),
                        ("idempotency key", build(True), "c0ffee-1042")):
    work_done.clear()
    statuses = client_with_retries(app, key)
    print(f"  {label:<18} responses {sorted(statuses)}   work done {len(work_done)} time(s)")

print("\nThree attempts at one request. Without a key the server did the work three times;")
print("with one it did it once, told the overlapping retry to wait, and replayed the stored")
print("result to the late one. The key is chosen by the client, once per logical request,")
print("and reused on every retry of it — which is what makes the retry safe to send.")
