# timeout: 1800
# Uses clarity/v0_17/service.py — the engine behind FastAPI.
# The engine behind HTTP, answering and streaming.

import asyncio
import contextlib
import json
import sys
import threading
import time

import httpx
import uvicorn

sys.path.insert(0, "code")
from clarity.v0_17.service import build_app                    # noqa: E402

PORT = 8732
QUESTION = "What was revenue in 2024 Q3, and what caused the Midwest decline?"

app = build_app()
server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=PORT,
                                       log_level="error"))
threading.Thread(target=server.run, daemon=True).start()
base = f"http://127.0.0.1:{PORT}"
for _ in range(200):
    with contextlib.suppress(Exception):
        httpx.get(f"{base}/health", timeout=1)
        break
    time.sleep(0.05)

print("--- validation happens before anything expensive ---\n")
for label, payload in (("a two-character question", {"question": "hi"}),
                       ("max_steps out of range",
                        {"question": "What was revenue in 2024 Q3?",
                         "max_steps": 99})):
    reply = httpx.post(f"{base}/ask", json=payload, timeout=30)
    detail = reply.json().get("detail", [{}])[0]
    print(f"  {label:<28}{reply.status_code}  "
          f"{detail.get('msg', '')[:44]}")

print("\n--- POST /ask ---\n")
started = time.perf_counter()
reply = httpx.post(f"{base}/ask", json={"question": QUESTION}, timeout=180).json()
plain_total = time.perf_counter() - started
print(f"  {plain_total:5.1f}s to the first byte and the last, because they are the "
      f"same byte")
print(f"  tools: {', '.join(reply['tools_used'])}")
print(f"  {' '.join(reply['answer'].split())[:66]}")

print("\n--- POST /ask/stream ---\n")
events = []


async def stream() -> float:
    started = time.perf_counter()
    first = None
    async with httpx.AsyncClient(timeout=180) as client:
        async with client.stream("POST", f"{base}/ask/stream",
                                 json={"question": QUESTION}) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                at = time.perf_counter() - started
                if first is None:
                    first = at
                events.append((at, event))
    return first


first_at = asyncio.run(stream())
for at, event in events:
    detail = (event.get("tool") or
              (" ".join(event.get("text", "").split())[:44] if event.get("text")
               else ""))
    print(f"  {at:5.2f}s  {event['event']:<9}{detail}")

stream_total = events[-1][0]
json.dump({"plain_total": plain_total, "stream_first": first_at,
           "stream_total": stream_total,
           "events": [{"at": a, **e} for a, e in events]},
          open("code/25/_service.json", "w"), indent=1)

print()
print(f"  first byte at {first_at:.2f}s, last at {stream_total:.1f}s")
print(f"  the plain endpoint took {plain_total:.1f}s and showed nothing until the end")
print()
print(f"Streaming did not make the work faster — both answers took about the same")
print(f"time. What changed is that a person sees the system working at "
      f"{first_at:.2f}s instead")
print(f"of staring at a spinner for {plain_total:.0f} seconds, and §28.6 is about how "
      f"much that")
print("difference is worth.")
print()
print("Two details in that stream matter more than the mechanism.")
print()
print("The steps are events, not text fragments. A token stream is what a chat")
print("interface wants; a *step* stream is what an agent should emit, because 'searching")
print("the 2024 Q3 review' is information and a half-written sentence is not.")
print()
print("And the work runs on a thread while the endpoint stays async. The engine is")
print("synchronous and takes seconds — §25.7 is what happens when you forget that —")
print("so it goes to an executor and posts its progress back to the loop through a")
print("queue.")
