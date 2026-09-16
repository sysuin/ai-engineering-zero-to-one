# Streaming: the server sends the answer a piece at a time, as server-sent events, and
# the client handles each piece as it arrives. Chapter 4 does this with a model.

import json
import time

import requests

from meridian_api import serve

BASE = serve()

start, first_at, text = time.perf_counter(), None, ""
with requests.get(f"{BASE}/events", params={"count": 10, "interval": 0.2},
                  stream=True, timeout=5) as response:
    print("Content-Type:     ", response.headers["Content-Type"])
    print("Transfer-Encoding:", response.headers["Transfer-Encoding"])
    event = "message"
    for line in response.iter_lines(decode_unicode=True):
        if line.startswith("event:"):
            event = line.split(":", 1)[1].strip()
        elif line.startswith("data:"):
            if event == "done":
                break
            if first_at is None:
                first_at = time.perf_counter() - start
            text += json.loads(line.split(":", 1)[1])["delta"]
        elif line == "":
            event = "message"                     # a blank line ends one event

total = time.perf_counter() - start
print(f"first piece after {first_at:.2f} s; the whole answer after {total:.2f} s")
print("assembled:", text.strip())
