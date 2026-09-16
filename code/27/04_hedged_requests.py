# The slowest one request in a hundred sets the experience of anyone who makes a hundred
# requests. A gateway can send a second copy of a slow request and take whichever answers
# first. Simulated latencies, so the trade can be counted exactly.

import numpy as np

rng = np.random.default_rng(27)
N = 200_000
# Most calls take about a second; a small share get stuck behind something and take many.
normal = rng.lognormal(np.log(1.0), 0.3, N)
stuck = rng.random(N) < 0.02
first = np.where(stuck, normal + rng.uniform(5, 20, N), normal)
second = np.where(rng.random(N) < 0.02, rng.lognormal(np.log(1.0), 0.3, N) + rng.uniform(5, 20, N),
                  rng.lognormal(np.log(1.0), 0.3, N))


def summary(latency: np.ndarray) -> str:
    p50, p95, p99 = np.percentile(latency, [50, 95, 99])
    return f"p50 {p50:4.2f}s   p95 {p95:5.2f}s   p99 {p99:5.2f}s"


print(f"  {'no hedging':30} {summary(first)}   extra calls   0.0%")
for hedge_at in (np.percentile(first, 90), np.percentile(first, 95)):
    sent_second = first > hedge_at
    # The second copy starts at hedge_at; whichever finishes first wins.
    hedged = np.where(sent_second, np.minimum(first, hedge_at + second), first)
    print(f"  {'hedge after ' + format(hedge_at, '.2f') + 's':30} {summary(hedged)}   "
          f"extra calls {sent_second.mean():5.1%}")
