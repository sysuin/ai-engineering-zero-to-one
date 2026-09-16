# How far does fan-out go? Fit Amdahl's law to the measured speed-up, then ask what
# more workers would buy — and what a provider's rate limit does to the answer.

import json

measured = json.load(open("code/20/_fanout.json"))
n = measured["n"]
speedup = measured["sequential"] / measured["parallel"]

# Amdahl: with a fraction s of the work serial, n workers give 1 / (s + (1 - s) / n).
serial = (n / speedup - 1) / (n - 1)
print(f"measured: {n} workers, {speedup:.1f}x faster")
print(f"Amdahl fit: {serial:.1%} of the work does not run in parallel")
print("(one measurement, so the fit is an illustration, not a model of your system)\n")

print(f"  {'workers':>7} {'speed-up':>9}")
for workers in (1, 6, 12, 24, 100, 1_000):
    predicted = 1 / (serial + (1 - serial) / workers)
    print(f"  {workers:>7,} {predicted:>8.1f}x")
print(f"\n  ceiling however many workers: {1 / serial:.0f}x" if serial > 0 else "")

# A rate limit caps concurrency before Amdahl does. With each task spending a known
# number of tokens per minute, the limit fixes how many can run at once.
per_task = measured["parallel_tokens"] / n
duration_min = measured["parallel"] / 60
tokens_per_minute_each = per_task / duration_min
for limit in (200_000, 2_000_000):                 # two illustrative limits, not a price list
    concurrent = int(limit // tokens_per_minute_each)
    print(f"\na limit of {limit:,} tokens per minute allows about {concurrent:,} of these "
          f"tasks at once")
