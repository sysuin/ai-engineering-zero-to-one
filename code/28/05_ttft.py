# timeout: 900
# Where the seconds go in one model call: the wait before the first token, which grows with
# the input, and the time after it, which grows with the output. Measured by streaming.

import statistics
import time
import uuid

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
PARAGRAPH = ("Meridian's quarterly reviews record revenue, margin, orders and supplier notes "
             "for every region, and the analyst reads them before answering. ")
MEDIUM, LONG = PARAGRAPH * 260, PARAGRAPH * 2_000                 # ~6,500 and ~50,000 tokens
SHORT_ANSWER = "Answer in one sentence of at most fifteen words."
LONG_ANSWER = "Answer in about three hundred words, in full paragraphs."
QUESTION = "Why might a distributor's margin fall in a quarter when revenue rises?"
REPEATS = 3


def timed(context: str, instruction: str) -> tuple[float, float, int, int, int]:
    # A fresh token at the very start defeats the provider's prefix cache, so a long input
    # is read from scratch every time rather than served from an earlier call.
    nonce = f"[request {uuid.uuid4().hex[:8]}]\n"
    started = time.perf_counter()
    first = None
    usage = None
    stream = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=1_200, stream=True,
        stream_options={"include_usage": True},
        messages=[{"role": "system", "content": nonce + context + instruction},
                  {"role": "user", "content": QUESTION}])
    for chunk in stream:
        if first is None and chunk.choices and chunk.choices[0].delta.content:
            first = time.perf_counter() - started
        if chunk.usage:
            usage = chunk.usage
    total = time.perf_counter() - started
    cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    return first, total, usage.prompt_tokens, usage.completion_tokens, cached


conditions = [("short input, short answer", "", SHORT_ANSWER),
              ("short input, long answer", "", LONG_ANSWER),
              ("6k input,    short answer", MEDIUM, SHORT_ANSWER),
              ("50k input,   short answer", LONG, SHORT_ANSWER)]
results = {}
print(f"{REPEATS} calls per condition; medians\n")
print(f"  {'':<27} {'in tok':>7} {'cached':>7} {'out tok':>8} {'first token':>12} "
      f"{'after it':>9}")
for label, context, instruction in conditions:
    runs = [timed(context, instruction) for _ in range(REPEATS)]
    first = statistics.median(r[0] for r in runs)
    after = statistics.median(r[1] - r[0] for r in runs)
    tokens_in = statistics.median(r[2] for r in runs)
    tokens_out = statistics.median(r[3] for r in runs)
    cached = statistics.median(r[4] for r in runs)
    results[label] = (first, after)
    print(f"  {label:<27} {tokens_in:>7,.0f} {cached:>7,.0f} {tokens_out:>8,.0f} "
          f"{first:>11.2f}s {after:>8.2f}s")

short_in = results["short input, short answer"][0]
mid_in = results["6k input,    short answer"][0]
long_in = results["50k input,   short answer"][0]
short_out = results["short input, short answer"][1]
long_out = results["short input, long answer"][1]
print(f"\nThe wait for the first token: {short_in:.2f}s with a short input, {mid_in:.2f}s at "
      f"about 6,500 tokens,")
print(f"{long_in:.2f}s at about 50,000. The time after it: {short_out:.2f}s for a short "
      f"answer, {long_out:.2f}s for a long one.")
grew_out = long_out / short_out
grew_in = long_in / short_in
print(f"\nOn this model and this afternoon, output length multiplied the second number by "
      f"{grew_out:.0f}x,")
print(f"and input length multiplied the first by {grew_in:.1f}x. Reading a prompt is fast next "
      "to writing an")
print("answer; it becomes the wait people notice only when prompts are very long, or")
print("when the provider is busy enough that queueing is added to it.")
