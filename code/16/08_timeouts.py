# A timeout that does not return on time. Leaving a `with ThreadPoolExecutor()`
# block waits for its threads — including one that .result(timeout=...) has
# already given up on.

import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout

TIMEOUT = 0.5


def hung_tool() -> str:
    time.sleep(3)          # a query with no index, a dead mount, a lock
    return "finished"


def with_block() -> str:
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(hung_tool).result(timeout=TIMEOUT)
    except FutureTimeout:
        return "timed out"  # reached only after the with-block has waited


def without_waiting() -> str:
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(hung_tool).result(timeout=TIMEOUT)
    except FutureTimeout:
        return "timed out"
    finally:
        pool.shutdown(wait=False)  # abandon the thread; it cannot be stopped


print(f"a tool that takes 3s, called with a {TIMEOUT}s timeout\n")
for label, attempt in (("with-block", with_block),
                       ("shutdown(wait=False)", without_waiting)):
    before = threading.active_count()
    started = time.perf_counter()
    outcome = attempt()
    elapsed = time.perf_counter() - started
    still = threading.active_count() - before
    print(f"  {label:21} {outcome!r} after {elapsed:.2f}s   "
          f"threads still running: {still}")

print("\nBoth report a timeout. Only one returned when it said it would.")
