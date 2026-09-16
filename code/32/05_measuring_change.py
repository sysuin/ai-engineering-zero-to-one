# Measuring a change on real traffic: how many requests a canary and an A/B test need,
# and what checking the result every day does to the answer.

import json
import math
from statistics import NormalDist

import numpy as np

z = NormalDist().inv_cdf
rng = np.random.default_rng(32)


def canary_requests(p0: float, p1: float, alpha=0.05, power=0.8) -> int:
    """Requests before a one-sided test can see a rate rise from p0 to p1."""
    top = z(1 - alpha) * math.sqrt(p0 * (1 - p0)) + z(power) * math.sqrt(p1 * (1 - p1))
    return math.ceil((top / (p1 - p0)) ** 2)


def ab_per_arm(p1: float, p2: float, alpha=0.05, power=0.8) -> int:
    """Observations per arm for a two-sided test of two proportions."""
    mean = (p1 + p2) / 2
    top = (z(1 - alpha / 2) * math.sqrt(2 * mean * (1 - mean))
           + z(power) * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2)))
    return math.ceil((top / (p2 - p1)) ** 2)


# ------------------------------------------------------------------ 1. a canary
REQUESTS_PER_DAY, CANARY_SHARE = 4_000, 0.05
print("1. A canary: requests needed to see a failure rate rise (80% power, 5% alpha)\n")
print(f"  {'from':>6} {'to':>6} {'requests':>9}   days at {CANARY_SHARE:.0%} of "
      f"{REQUESTS_PER_DAY:,}/day")
canary = []
for p0, p1 in ((0.01, 0.03), (0.01, 0.02), (0.05, 0.07), (0.01, 0.015)):
    n = canary_requests(p0, p1)
    days = n / (REQUESTS_PER_DAY * CANARY_SHARE)
    canary.append({"from": p0, "to": p1, "n": n, "days": days})
    print(f"  {p0:>6.1%} {p1:>6.1%} {n:>9,}   {days:>6.1f}")

# ------------------------------------------------------------------ 2. an A/B test
RATED_SHARE = 0.05                        # most people never press a feedback button
rated_per_arm_per_day = REQUESTS_PER_DAY * RATED_SHARE / 2
print(f"\n2. An A/B test on the thumbs-up rate, with {RATED_SHARE:.0%} of answers rated\n")
print(f"  {'baseline':>8} {'better':>7} {'ratings per arm':>16} {'days':>6}")
ab = []
for p1, p2 in ((0.60, 0.70), (0.60, 0.65), (0.60, 0.63)):
    n = ab_per_arm(p1, p2)
    ab.append({"from": p1, "to": p2, "n": n, "days": n / rated_per_arm_per_day})
    print(f"  {p1:>8.0%} {p2:>7.0%} {n:>16,} {n / rated_per_arm_per_day:>6.0f}")

# ------------------------------------------------------------------ 3. peeking
DAYS, PER_DAY, TRIALS, RATE = 20, 100, 4_000, 0.60
print(f"\n3. Two identical arms, checked every day for {DAYS} days "
      f"({PER_DAY} ratings per arm per day)\n")
a = rng.binomial(PER_DAY, RATE, (TRIALS, DAYS)).cumsum(axis=1)
b = rng.binomial(PER_DAY, RATE, (TRIALS, DAYS)).cumsum(axis=1)
n = PER_DAY * np.arange(1, DAYS + 1)
pooled = (a + b) / (2 * n)
zscore = (a - b) / n / np.sqrt(pooled * (1 - pooled) * 2 / n)
significant = np.abs(zscore) > z(0.975)
once_at_end = significant[:, -1].mean()
ever = significant.any(axis=1).mean()
print(f"  looked once, at the end         {once_at_end:>5.1%} declared a difference")
print(f"  looked daily, stopped at p<0.05 {ever:>5.1%} declared a difference")
print(f"\n  Neither arm was better. Checking daily and stopping at the first significant")
print(f"  day turned a {once_at_end:.0%} false-positive rate into {ever:.0%}.")

json.dump({"canary": canary, "ab": ab, "peeking": {"end": once_at_end, "daily": ever}},
          open("code/32/_measuring_change.json", "w"), indent=1)
