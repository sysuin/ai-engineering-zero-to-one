# timeout: 600
# Two ways to load-test the same service. A closed-loop test sends the next request when the
# last one returns; an open-loop test sends requests on a schedule, as real users do. Only
# one of them can show you what happens past capacity.

import asyncio
import contextlib
import socket
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx
import uvicorn
from fastapi import FastAPI

WORKERS, WORK = 8, 0.25                     # capacity: 8 / 0.25 = 32 requests a second
DURATION = 8.0


@contextlib.asynccontextmanager
async def bounded_pool(_: FastAPI):
    asyncio.get_running_loop().set_default_executor(ThreadPoolExecutor(WORKERS))
    yield


app = FastAPI(lifespan=bounded_pool)


@app.get("/ask")
async def ask() -> dict:
    await asyncio.to_thread(time.sleep, WORK)          # stands in for the model calls
    return {"ok": True}


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


PORT = free_port()
server = uvicorn.Server(uvicorn.Config(app, port=PORT, log_level="warning"))
threading.Thread(target=server.run, daemon=True).start()
while not server.started:
    time.sleep(0.05)
URL = f"http://127.0.0.1:{PORT}/ask"


async def closed_loop(users: int) -> tuple[float, list[tuple[float, float]]]:
    samples: list[tuple[float, float]] = []            # (when it was due, how long it took)
    start = time.perf_counter()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1_000)) as client:
        async def user():
            while time.perf_counter() - start < DURATION:
                sent = time.perf_counter()
                await client.get(URL, timeout=60)
                samples.append((sent - start, time.perf_counter() - sent))
        await asyncio.gather(*(user() for _ in range(users)))
    served = sum(1 for due, took in samples if due + took <= DURATION)
    return served / DURATION, samples


async def open_loop(rate: float) -> tuple[float, list[tuple[float, float]]]:
    samples: list[tuple[float, float]] = []
    start = time.perf_counter()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=1_000)) as client:
        async def one(due: float):
            await asyncio.sleep(max(0.0, due - (time.perf_counter() - start)))
            await client.get(URL, timeout=60)
            # Measured from when the request was *due*, not from when it was sent: a user
            # does not wait politely for the previous request to finish before arriving.
            samples.append((due, time.perf_counter() - start - due))
        await asyncio.gather(*(one(i / rate) for i in range(int(rate * DURATION))))
    served = sum(1 for due, took in samples if due + took <= DURATION)
    return served / DURATION, samples


def halves(samples: list[tuple[float, float]]) -> tuple[float, float]:
    early = [took for due, took in samples if due < DURATION / 2]
    late = [took for due, took in samples if due >= DURATION / 2]
    return statistics.median(early), statistics.median(late)


print(f"A service with {WORKERS} workers and {WORK}s of work per request: "
      f"capacity {WORKERS / WORK:.0f} requests a second.\n")
print(f"  {'':<24} {'served/s':>9} {'p50, first half':>16} {'p50, second half':>17}")
results = {}
for label, test in (("closed loop,  8 users", closed_loop(8)),
                    ("closed loop, 64 users", closed_loop(64)),
                    ("open loop, 30 req/s", open_loop(30)),
                    ("open loop, 36 req/s", open_loop(36))):
    served, samples = asyncio.run(test)
    early, late = halves(samples)
    results[label] = (served, early, late)
    print(f"  {label:<24} {served:>9.1f} {early:>15.2f}s {late:>16.2f}s")

server.should_exit = True
closed = results["closed loop, 64 users"]
opened = results["open loop, 36 req/s"]
print(f"\nSixty-four closed-loop users push the service past capacity, and the test reports a")
print(f"steady state: {closed[1]:.2f}s early, {closed[2]:.2f}s late. It cannot report anything else,")
print("because each user waits for an answer before asking again, so the queue never grows")
print("longer than the number of users. Open-loop arrivals just above capacity do what real")
print(f"users do: the median wait went from {opened[1]:.2f}s in the first half to {opened[2]:.2f}s in the")
print("second, and it keeps growing for as long as the arrivals continue. A load test that")
print("cannot show that has been coordinated with the thing it was meant to stress.")
