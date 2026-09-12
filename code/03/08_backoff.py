# Exponential backoff: wait longer after each failure, and honour Retry-After.

import random
import time

import requests

from meridian_api import serve

BASE = serve()
RETRYABLE = {408, 425, 429, 500, 502, 503, 504}


def get_with_backoff(url: str, *, attempts: int = 6, base: float = 0.25,
                     cap: float = 8.0, jitter: bool = True) -> dict:
    """
    GET `url`, retrying only what is worth retrying.

    The wait doubles each time: base, 2*base, 4*base ... up to `cap`. With jitter on,
    the actual wait is a random point between zero and that ceiling.
    """
    for attempt in range(attempts):
        response = requests.get(url, timeout=10)

        if response.status_code < 400:
            return response.json()
        if response.status_code not in RETRYABLE:
            raise RuntimeError(f"not retryable: {response.status_code}")
        if attempt == attempts - 1:
            break

        ceiling = min(cap, base * (2 ** attempt))

        # A server that tells you when to come back has the better information.
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            wait = float(retry_after)
            why = f"Retry-After: {retry_after}s"
        else:
            wait = random.uniform(0, ceiling) if jitter else ceiling
            why = f"ceiling {ceiling:.2f}s"

        print(f"  attempt {attempt + 1}: {response.status_code}, "
              f"waiting {wait:.2f}s  ({why})")
        time.sleep(wait)

    raise RuntimeError(f"gave up after {attempts} attempts")


random.seed(7)      # so this listing prints the same numbers every time

print("A server that fails four times:")
print("  ->", get_with_backoff(f"{BASE}/flaky?key=backoff&fail=4"))

print("\nA server that tells you to slow down:")
try:
    get_with_backoff(f"{BASE}/ratelimited", attempts=3)
except RuntimeError as error:
    print("  ->", error)
