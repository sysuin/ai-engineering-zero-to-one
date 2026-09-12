# Retrying, the obvious way: try again after a fixed pause.

import time

import requests

from meridian_api import serve

BASE = serve()

def fetch(url: str, attempts: int = 5, delay: float = 0.4) -> dict:
    """Try `attempts` times, pausing `delay` seconds between tries."""
    for attempt in range(1, attempts + 1):
        response = requests.get(url, timeout=10)
        print(f"  attempt {attempt}: {response.status_code}")

        if response.status_code < 400:
            return response.json()

        if 400 <= response.status_code < 500 and response.status_code != 429:
            # You sent something wrong. Trying again will send the same wrong thing.
            raise RuntimeError(f"not retryable: {response.status_code}")

        time.sleep(delay)

    raise RuntimeError(f"gave up after {attempts} attempts")


print("Retrying a server that fails twice, then works:")
result = fetch(f"{BASE}/flaky?key=naive&fail=2")
print("  ->", result)

print("\nRetrying a request that is simply wrong:")
try:
    fetch(f"{BASE}/revenue")
except RuntimeError as error:
    print("  ->", error)
