# An exact cache hits only when a question has been asked before, recently enough, since the
# last time the cache was emptied. How often that happens is set by how concentrated the traffic
# is, which is a property of the users. Simulated for traffic whose popularity follows a power
# law, the shape query logs usually show.

import numpy as np

rng = np.random.default_rng(28)
DISTINCT, PER_DAY, DAYS = 20_000, 5_000, 14


def traffic(s: float) -> tuple[np.ndarray, np.ndarray]:
    """Question ids and arrival times (hours) for DAYS of traffic; rank r asked in proportion
    to 1 / r**s."""
    weights = 1 / np.arange(1, DISTINCT + 1) ** s
    n = PER_DAY * DAYS
    ids = rng.choice(DISTINCT, n, p=weights / weights.sum())
    hours = np.sort(rng.uniform(0, 24 * DAYS, n))
    return ids, hours


def hit_rate(ids: np.ndarray, hours: np.ndarray, ttl: float, flush_daily: bool) -> float:
    """Share of requests in the second week answered from cache."""
    last: dict[int, float] = {}
    hits = counted = 0
    for q, t in zip(ids.tolist(), hours.tolist()):
        seen = last.get(q)
        fresh = seen is not None and t - seen <= ttl
        if flush_daily and seen is not None and seen // 24 != t // 24:
            fresh = False            # the nightly rebuild emptied the cache
        if t >= 24 * 7:
            counted += 1
            hits += fresh
        if not fresh:
            last[q] = t              # a miss is answered and stored
    return hits / counted


print(f"{PER_DAY:,} requests a day drawn from {DISTINCT:,} distinct questions;")
print("hit rate over the second week\n")
print(f"  {'exponent':>9}{'top 1%':>8}{'time to live':^33}{'1 day, flushed':>16}")
print(f"  {'s':>9}{'share':>8}{'1 hour':>11}{'1 day':>11}{'1 week':>11}{'nightly':>16}")
for s in (0.6, 0.8, 1.0, 1.2):
    ids, hours = traffic(s)
    weights = 1 / np.arange(1, DISTINCT + 1) ** s
    top = weights[: DISTINCT // 100].sum() / weights.sum()
    cells = "".join(f"{hit_rate(ids, hours, ttl, False):>11.0%}" for ttl in (1, 24, 24 * 7))
    print(f"  {s:>9}{top:>8.0%}{cells}{hit_rate(ids, hours, 24, True):>16.0%}")
print("\ntop 1% share: the share of all requests that ask one of the most common")
print("1% of questions")
