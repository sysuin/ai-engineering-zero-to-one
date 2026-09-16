# The fix for a blocked loop is a thread. Threads come from a pool, and the default pool
# has a size. Forty slow requests, each pushed to a thread the standard way.

import asyncio
import os
import time
from concurrent.futures import ThreadPoolExecutor

REQUESTS, WORK = 40, 0.5
default_workers = ThreadPoolExecutor()._max_workers    # what asyncio creates when none is set


async def serve(executor: ThreadPoolExecutor | None) -> float:
    loop = asyncio.get_running_loop()
    if executor is not None:
        loop.set_default_executor(executor)
    started = time.perf_counter()
    await asyncio.gather(*(asyncio.to_thread(time.sleep, WORK) for _ in range(REQUESTS)))
    return time.perf_counter() - started


print(f"{REQUESTS} requests, each blocking for {WORK}s in a thread; "
      f"this machine has {os.cpu_count()} CPUs\n")
for label, executor in ((f"default pool ({default_workers} threads)", None),
                        (f"a pool of {REQUESTS} threads", ThreadPoolExecutor(REQUESTS))):
    took = asyncio.run(serve(executor))
    print(f"  {label:28} {took:5.2f}s   ({took / WORK:.0f}x one request)")
