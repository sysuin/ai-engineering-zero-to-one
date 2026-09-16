# Twelve independent calls, three ways. A model call is almost entirely waiting, and
# waiting is something a program can do several of at once.

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor

from openai import AsyncOpenAI, OpenAI

from clarity.config import MODEL_FAST

REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest", "Central",
           "Pacific", "Mountain", "Atlantic", "Gulf", "Plains", "Lakes"]
WORKERS = 6


def prompt(region: str) -> list[dict]:
    return [{"role": "user", "content": f"Suggest a four-word name for a {region} "
                                        "sales team. Reply with the name only."}]


client = OpenAI()


def one(region: str) -> float:
    started = time.perf_counter()
    client.chat.completions.create(model=MODEL_FAST, messages=prompt(region),
                                   max_completion_tokens=20)
    return time.perf_counter() - started


# 1. One after another.
started = time.perf_counter()
latencies = [one(r) for r in REGIONS]
sequential = time.perf_counter() - started

# 2. A pool of threads, each blocking on its own call.
started = time.perf_counter()
with ThreadPoolExecutor(max_workers=WORKERS) as pool:
    list(pool.map(one, REGIONS))
threaded = time.perf_counter() - started


# 3. asyncio: one thread, many calls waiting at once, a semaphore capping how many.
async def main() -> float:
    aclient = AsyncOpenAI()
    gate = asyncio.Semaphore(WORKERS)

    async def one_async(region: str) -> None:
        async with gate:
            await aclient.chat.completions.create(model=MODEL_FAST, messages=prompt(region),
                                                  max_completion_tokens=20)

    started = time.perf_counter()
    await asyncio.gather(*(one_async(r) for r in REGIONS))
    return time.perf_counter() - started

asynchronous = asyncio.run(main())

print(f"{len(REGIONS)} calls; slowest single call {max(latencies):.2f}s, "
      f"sum of all calls {sum(latencies):.2f}s\n")
print(f"  sequential              {sequential:5.2f}s")
print(f"  {WORKERS} threads               {threaded:5.2f}s   "
      f"{sequential / threaded:.1f}x faster")
print(f"  asyncio, {WORKERS} at a time      {asynchronous:5.2f}s   "
      f"{sequential / asynchronous:.1f}x faster")
print(f"\nThe ceiling is about {WORKERS}x with {WORKERS} calls in flight. Each run makes its own calls")
print("with their own latencies, so a measurement can land a little either side of it.")
print("Rate limits set a lower ceiling still.")
