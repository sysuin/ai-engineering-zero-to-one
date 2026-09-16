# The library retries and times out on your behalf. Worth knowing what it does by default,
# because a quick failure and a slow one can be the same failure.

import time

import httpx
import openai
from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
print("default timeout:    ", client.timeout)
print("default max_retries:", client.max_retries)

attempts = 0


def count(request):
    global attempts
    attempts += 1


GOOD = [{"role": "user", "content": "Say OK."}]
for retries in (2, 0):
    attempts = 0
    counted = OpenAI(max_retries=retries, timeout=0.05,        # far too short, on purpose
                     http_client=httpx.Client(event_hooks={"request": [count]}))
    started = time.perf_counter()
    try:
        counted.chat.completions.create(model=MODEL_FAST, messages=GOOD)
        outcome = "succeeded"
    except openai.APITimeoutError:
        outcome = "APITimeoutError"
    print(f"\nmax_retries={retries}: {outcome} after {time.perf_counter() - started:.2f}s "
          f"and {attempts} HTTP attempt{'s' if attempts != 1 else ''}")

print("\nThe retries were silent. With max_retries=2 the same error took several times as")
print("long to arrive. Chapter 26 decides where retries belong; the first step is to know")
print("that the library is already doing some.")
