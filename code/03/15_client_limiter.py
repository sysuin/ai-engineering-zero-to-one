# The polite client: pace yourself with your own token bucket, and the server's never
# runs dry. Same 40 requests, sent two ways, to a server that allows 5 per second.

import threading
import time

import requests

from meridian_api import serve

BASE = serve()
N = 40


class TokenBucket:
    """Blocks until a request is allowed: `rate` per second, bursts up to `capacity`."""

    def __init__(self, rate: float, capacity: int):
        self.rate, self.capacity = rate, capacity
        self.tokens, self.at = float(capacity), time.monotonic()
        self.lock = threading.Lock()

    def acquire(self):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.capacity, self.tokens + (now - self.at) * self.rate)
                self.at = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                wait = (1 - self.tokens) / self.rate
            time.sleep(wait)


def run(label, before_each=None):
    ok = limited = 0
    start = time.perf_counter()
    for _ in range(N):
        if before_each:
            before_each()
        status = requests.get(f"{BASE}/limited", timeout=5).status_code
        ok += status == 200
        limited += status == 429
    print(f"  {label:24} {ok:3} succeeded  {limited:3} got 429   "
          f"{time.perf_counter() - start:5.1f} s")
    time.sleep(1.2)                               # let the server's bucket refill


print(f"{N} requests to a server that allows 5 per second (burst of 5)")
run("as fast as possible")
bucket = TokenBucket(rate=4.5, capacity=1)        # a little under the server's rate
run("paced by a token bucket", bucket.acquire)
