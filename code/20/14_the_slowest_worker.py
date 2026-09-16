# timeout: 900
# A fan-out that waits for every worker is as slow as its slowest one. 240 identical model calls
# are timed, and fan-outs of different widths are then drawn from those measured times.

import json
import random
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                    # noqa: E402

client = OpenAI()
CALLS, DRAWS = 240, 20_000
PROMPT = ("Midwest revenue was $1,165,398 and West revenue was $1,402,117. "
          "Which region was lower? Answer with the region only.")


def one_call(_) -> float:
    started = time.perf_counter()
    client.chat.completions.create(model=MODEL_FAST, max_completion_tokens=400,
                                   messages=[{"role": "user", "content": PROMPT}])
    return time.perf_counter() - started


with ThreadPoolExecutor(max_workers=8) as pool:
    seconds = sorted(pool.map(one_call, range(CALLS)))
json.dump(seconds, open("code/20/_worker_seconds.json", "w"))


def pct(values: list[float], p: float) -> float:
    return values[min(len(values) - 1, int(p / 100 * len(values)))]


p90, p95 = pct(seconds, 90), pct(seconds, 95)
print(f"{CALLS} identical calls: median {statistics.median(seconds):.2f}s, p90 {p90:.2f}s, "
      f"p95 {p95:.2f}s, slowest {seconds[-1]:.2f}s\n")
rng = random.Random(20)


def fan_out(width: int, rule: str) -> float:
    """Seconds until the orchestrator can go on, for one fan-out."""
    times = [rng.choice(seconds) for _ in range(width)]
    if rule == "hedge at p90":           # a second copy of a straggler
        times = [t if t <= p90 else min(t, p90 + rng.choice(seconds))
                 for t in times]
    if rule == "all but the slowest" and width > 1:
        return sorted(times)[-2]
    return max(times)


print(f"  {'workers':>7}  {'wait for all':>19}  {'slower than p95':>15}  {'hedge at p90':>12}"
      f"  {'all but slowest':>16}")
print(f"  {'':>7}  {'median':>9}{'p95':>10}  {'':>15}  {'p95':>12}  {'p95':>16}")
for width in (1, 2, 6, 12, 24):
    plain = sorted(fan_out(width, "all") for _ in range(DRAWS))
    hedged = sorted(fan_out(width, "hedge at p90") for _ in range(DRAWS))
    partial = sorted(fan_out(width, "all but the slowest") for _ in range(DRAWS))
    slow = sum(t > p95 for t in plain) / DRAWS
    print(f"  {width:>7}  {statistics.median(plain):>8.2f}s{pct(plain, 95):>9.2f}s  {slow:>15.0%}"
          f"  {pct(hedged, 95):>11.2f}s  {pct(partial, 95):>15.2f}s")
print("\n  slower than p95 = share of fan-outs that took longer than one call's p95;")
print(f"  for independent calls it is 1 - 0.95^n: "
      + ", ".join(f"{1 - 0.95 ** n:.0%}" for n in (1, 2, 6, 12, 24)))
