# timeout: 1200
# The number one async bug, measured on ten concurrent requests.

import asyncio
import json
import time

from fastapi import FastAPI

CONCURRENT = 10
WORK_SECONDS = 0.4          # stands in for waiting on a model

app = FastAPI()


@app.get("/blocking")
async def blocking() -> dict:
    """
    An async endpoint that calls something synchronous.

    This is the shape of nearly every real occurrence: `async def` on the outside,
    and inside it a client library that was not written for asyncio. It looks
    concurrent. It is not.
    """
    time.sleep(WORK_SECONDS)
    return {"ok": True}


@app.get("/threaded")
async def threaded() -> dict:
    """The same synchronous work, moved off the event loop."""
    await asyncio.to_thread(time.sleep, WORK_SECONDS)
    return {"ok": True}


@app.get("/native")
async def native() -> dict:
    """Genuinely async all the way down — what you get with an async client."""
    await asyncio.sleep(WORK_SECONDS)
    return {"ok": True}


# A real server in one process with one event loop. TestClient will not do: it gives
# each request its own loop, so the bug this listing is about cannot happen there —
# which is a useful warning about what a test client does and does not prove.
import contextlib                                              # noqa: E402
import threading                                               # noqa: E402

import httpx                                                   # noqa: E402
import uvicorn                                                 # noqa: E402

PORT = 8731
config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
server = uvicorn.Server(config)
thread = threading.Thread(target=server.run, daemon=True)
thread.start()
for _ in range(200):
    with contextlib.suppress(Exception):
        httpx.get(f"http://127.0.0.1:{PORT}/native", timeout=1)
        break
    time.sleep(0.05)


def hammer(path: str) -> float:
    """Ten requests at once, against a single-process server with one event loop."""
    async def go():
        async with httpx.AsyncClient(timeout=60) as client:
            started = time.perf_counter()
            await asyncio.gather(*(client.get(f"http://127.0.0.1:{PORT}{path}")
                                   for _ in range(CONCURRENT)))
            return time.perf_counter() - started
    return asyncio.run(go())


ideal = WORK_SECONDS
serial = WORK_SECONDS * CONCURRENT
print(f"{CONCURRENT} concurrent requests, each doing {WORK_SECONDS}s of waiting.\n")
print(f"  perfectly concurrent would be {ideal:.1f}s; fully serial would be "
      f"{serial:.1f}s.\n")

results = {}
for label, path in (("async def + time.sleep()", "/blocking"),
                    ("asyncio.to_thread()", "/threaded"),
                    ("asyncio.sleep()", "/native")):
    elapsed = hammer(path)
    results[label] = elapsed
    bar = "#" * max(1, round(elapsed / serial * 40))
    print(f"  {label:<28}{elapsed:>6.2f}s  {bar}")

json.dump({"concurrent": CONCURRENT, "work": WORK_SECONDS, "serial": serial,
           "results": results}, open("code/25/_blocking.json", "w"), indent=1)

blocked = results["async def + time.sleep()"]
best = min(results.values())
print()
print(f"  the blocking version is {blocked / best:.0f}x slower than the other two,")
print(f"  and within {abs(blocked - serial) / serial:.0%} of doing the work one "
      f"request at a time")
print()
print("The `async def` did nothing. An event loop is cooperative: a coroutine keeps")
print("the thread until it awaits something, and `time.sleep` never awaits. Ten")
print("requests arrive, the first one seizes the loop for four hundred milliseconds,")
print("and the other nine sit in a queue that no dashboard calls a queue.")
print()
print("What makes this the number one async bug is not the mechanism, it is the")
print("symptom. Throughput collapses and every single request still looks fast in the")
print("logs: each one took 0.4s of work, and the nine that waited do not record what")
print("they were waiting for. The latency is real and lives nowhere you are looking —")
print("which is exactly the sort of thing §23.2 says to read a trace for.")
print()
print("Three ways to be wrong about this, in order of how often they happen:")
print()
print("  a sync client in an async endpoint    the case above; wrap it in to_thread")
print("  a CPU-bound step anywhere at all      tokenising, parsing a large document,")
print("                                        an embedding computed locally")
print("  a lock or a file read you forgot      logging to a file with a sync handler")
print("                                        does this under load")
print()
print("And the way to find it: if p95 degrades as concurrency rises while the work")
print("per request stays flat, you are looking at a blocked loop, not a slow model.")
