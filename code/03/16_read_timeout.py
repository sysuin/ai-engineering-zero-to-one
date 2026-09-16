# `timeout=` is not a deadline. The read timeout limits the gap between bytes, so a
# server that sends a byte every half-second never trips a one-second timeout.

import time

import requests

from meridian_api import serve

BASE = serve()

start = time.perf_counter()
response = requests.get(f"{BASE}/trickle", params={"chunks": 8, "interval": 0.5},
                        timeout=1.0)
print(f"timeout=1.0, status {response.status_code}, took {time.perf_counter() - start:.1f} s"
      " — and no Timeout was raised")

# (connect, read) as a pair: fail fast if the server is unreachable, be patient reading.
start = time.perf_counter()
try:
    requests.get("http://10.255.255.1/", timeout=(0.5, 30))
except requests.exceptions.ConnectTimeout:
    print(f"connect timeout after {time.perf_counter() - start:.1f} s to an unroutable address")
except requests.exceptions.ConnectionError as error:
    print(f"connection refused or unreachable: {type(error).__name__}")

# A real deadline: stream the body and check the clock yourself.
DEADLINE = 2.0
start = time.perf_counter()
received = b""
try:
    with requests.get(f"{BASE}/trickle", params={"chunks": 8, "interval": 0.5},
                      timeout=1.0, stream=True) as r:
        for chunk in r.iter_content(chunk_size=1):
            received += chunk
            if time.perf_counter() - start > DEADLINE:
                raise TimeoutError(f"deadline of {DEADLINE} s passed")
except TimeoutError as error:
    print(f"stopped: {error}, after {len(received)} of 8 bytes")
