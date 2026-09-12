# Streaming: the same answer, but you start seeing it much sooner.

import time

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
PROMPT = "In about 90 words, explain what a quarterly business review is for."

# Waiting for the whole thing.
started = time.time()
response = client.chat.completions.create(
    model=MODEL_FAST, messages=[{"role": "user", "content": PROMPT}])
blocking_total = time.time() - started
blocking_text = response.choices[0].message.content

print(f"Blocking:   nothing for {blocking_total:.2f}s, then {len(blocking_text)} "
      f"characters at once")

# Taking it as it arrives.
started = time.time()
first_token_at = None
pieces = []

stream = client.chat.completions.create(
    model=MODEL_FAST, messages=[{"role": "user", "content": PROMPT}], stream=True)

for chunk in stream:
    piece = chunk.choices[0].delta.content
    if piece:
        if first_token_at is None:
            first_token_at = time.time() - started
        pieces.append(piece)

streaming_total = time.time() - started
streamed_text = "".join(pieces)

print(f"Streaming:  first characters after {first_token_at:.2f}s, "
      f"all {len(streamed_text)} after {streaming_total:.2f}s")
print()
print(f"Time before the reader sees anything: "
      f"{blocking_total:.2f}s vs {first_token_at:.2f}s "
      f"— {blocking_total / first_token_at:.1f}x better")
print()
print("Total time is roughly the same. What changes is the waiting, and waiting is")
print("the part users experience. Chapter 28 treats this as a latency technique.")
