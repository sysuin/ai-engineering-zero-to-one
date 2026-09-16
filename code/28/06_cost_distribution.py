# Depends on code/32/_card.json; re-run after Chapter 32's numbers card changes.
# Cost per request is a distribution, not a number. The 120 answers behind Chapter 32's
# numbers card, read as a spread of tokens rather than an average.

import json
import statistics

import numpy as np

rows = json.load(open("code/32/_card.json"))["rows"]
tokens = np.array(sorted(r["tokens"] for r in rows))
total = tokens.sum()

print(f"{len(tokens)} answers from one run of Clarity v1.0 — the tokens each agent loop reported,")
print("not counting the model calls its tools make inside them\n")
print(f"  mean     {tokens.mean():>7,.0f} tokens")
for q in (50, 90, 99):
    print(f"  p{q:<6} {np.percentile(tokens, q):>7,.0f} tokens")
print(f"  largest  {tokens.max():>7,} tokens   "
      f"({tokens.max() / np.median(tokens):.1f}x the median)")

top = tokens[::-1][: len(tokens) // 10]
print(f"\n  the most expensive tenth of answers used {top.sum() / total:.0%} of all tokens")

by_kind: dict[str, list[int]] = {}
for r in rows:
    by_kind.setdefault(r["kind"], []).append(r["tokens"])
print(f"\n  {'population':<14} {'n':>4} {'median':>8} {'p90':>8} {'share of tokens':>16}")
for kind, values in sorted(by_kind.items(), key=lambda kv: -sum(kv[1])):
    print(f"  {kind:<14} {len(values):>4} {statistics.median(values):>8,.0f} "
          f"{np.percentile(values, 90):>8,.0f} {sum(values) / total:>16.0%}")

mean, median = tokens.mean(), np.median(tokens)
costliest = max(by_kind, key=lambda k: statistics.median(by_kind[k]))
share_n = len(by_kind[costliest]) / len(rows)
share_t = sum(by_kind[costliest]) / total
print(f"\nThe mean is {mean / median:.1f}x the median: a few long runs pull it up, so a budget")
print("set from the mean is too generous for most requests and too tight for the ones")
print("that need it. Budget per request from a high percentile, and look at who lives there.")
print(f"\nHere it is the {costliest} questions: {share_n:.0%} of the requests and "
      f"{share_t:.0%} of the tokens.")
if costliest == "unanswerable":
    print("An agent searches hardest for what is not there, so the cheapest refusal is the")
    print("one decided early — which is a cost argument for §13.11's abstention, not only a")
    print("quality one.")
