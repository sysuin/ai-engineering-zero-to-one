# What the time is made of. Latency is roughly a fixed wait before the first token, plus
# a steady cost for every token after it — so the length of the answer decides most of it.

import statistics
import time

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
TOPIC = "the history of the shipping container"


def timed(words: int) -> tuple[float, float, int]:
    started, first = time.perf_counter(), None
    tokens = 0
    stream = client.chat.completions.create(
        model=MODEL_FAST, stream=True, stream_options={"include_usage": True},
        messages=[{"role": "user", "content": f"Write about {words} words on {TOPIC}."}])
    for chunk in stream:
        if chunk.usage is not None:
            tokens = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content and first is None:
            first = time.perf_counter() - started
    return first, time.perf_counter() - started, tokens


rows = []
print(f"  {'asked for':>9}  {'tokens':>6}  {'first token':>11}  {'total':>6}  {'per token':>9}")
for words in (25, 100, 300):
    for _ in range(2):
        first, total, tokens = timed(words)
        per = (total - first) / max(tokens - 1, 1)
        rows.append((tokens, first, total, per))
        print(f"  {words:>6} w  {tokens:>6}  {first:>10.2f}s  {total:>5.2f}s  "
              f"{1000 * per:>7.1f}ms")

firsts = [r[1] for r in rows]
pers = [r[3] for r in rows]
short = statistics.mean(r[2] for r in rows[:2])
long_ = statistics.mean(r[2] for r in rows[-2:])
print(f"\nmedian time to first token {statistics.median(firsts):.2f}s; "
      f"median time per output token {1000 * statistics.median(pers):.1f}ms")
print(f"the longest answers took {long_ / short:.1f}x as long as the shortest, "
      f"while the first token arrived in about the same time")
