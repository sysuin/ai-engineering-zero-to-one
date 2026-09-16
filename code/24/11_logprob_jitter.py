# timeout: 900
# Why temperature 0 still diverges, seen from inside: the same request, thirty times, with the
# model's log-probabilities returned. How much do the numbers for the same token at the same
# position move between calls, and how close were the two best tokens where the runs split?

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                          # noqa: E402
from openai import OpenAI                                      # noqa: E402

RUNS, TOKENS = 30, 120
client = OpenAI()
QUESTION = ("A supplier may raise prices once in any twelve month period by no more "
            "than 3%, on 30 days notice. They raised prices 2.8% on 25 days notice, "
            "twice in one year. List every clause breached, most serious first.")


def run(_):
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=TOKENS, logprobs=True,
        top_logprobs=2, messages=[{"role": "user", "content": QUESTION}])
    reasoning = reply.usage.completion_tokens_details.reasoning_tokens or 0
    tokens = [(t.token, t.logprob,
               t.top_logprobs[1].logprob if len(t.top_logprobs) > 1 else None,
               t.top_logprobs[0].token == t.token)
              for t in reply.choices[0].logprobs.content]
    return tokens, reasoning


with ThreadPoolExecutor(max_workers=10) as pool:
    results = list(pool.map(run, range(RUNS)))
runs = [tokens for tokens, _ in results]

from collections import Counter

print(f"{RUNS} identical requests at temperature 0, log-probabilities returned")
print(f"reasoning tokens in any run: {sum(r for _, r in results)}; runs whose first token was not "
      f"their own top choice: {sum(not r[0][3] for r in runs)}\n")
first = Counter(r[0][0] for r in runs)
print("  the first token chosen: " + ", ".join(f"{tok!r} {n}x" for tok, n in first.most_common()))

# Follow the most common path: at each position, the runs that have produced exactly that path
# so far, the log-probability each gave the next token on it, and the gap to the runner-up.
print(f"\n  {'position':>8}  {'token':12}{'runs on path':>13}{'log-prob, lowest to highest':>30}{'gap to 2nd':>12}")
path = [r for r in runs]
for position in range(6):
    token, _ = Counter(r[position][0] for r in path).most_common(1)[0]
    on = [r for r in path if r[position][0] == token]
    values = [r[position][1] for r in on]
    gaps = [r[position][1] - r[position][2] for r in on if r[position][2] is not None]
    gap = f"{np.median(gaps):.2f}" if gaps else "-"
    print(f"  {position + 1:>8}  {token!r:12}{len(on):>13}{min(values):>15.4f} to {max(values):<11.4f}{gap:>12}")
    path = on

lowest, highest = min(r[0][1] for r in runs if r[0][0] == first.most_common(1)[0][0]), \
    max(r[0][1] for r in runs if r[0][0] == first.most_common(1)[0][0])
print(f"\nthe same first token, from the same request, was given probabilities from "
      f"{np.exp(lowest):.0%} to {np.exp(highest):.0%}")
