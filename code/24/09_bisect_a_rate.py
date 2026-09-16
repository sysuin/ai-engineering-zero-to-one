# git bisect on a probabilistic bug. Commit 41 of 64 raised a failure rate from 5% to 30%;
# the bisect script runs the case n times and calls the commit bad above a threshold. How often
# does the search end at commit 41?

import math
import random

COMMITS, CULPRIT = 64, 41
RATE_BEFORE, RATE_AFTER = 0.05, 0.30
TRIALS = 20_000


def bisect(runs: int, threshold: int, rng: random.Random) -> tuple[int, int]:
    good, bad, tests = 0, COMMITS, 0                   # commit 0 is known good, the last known bad
    while bad - good > 1:
        mid = (good + bad) // 2
        rate = RATE_AFTER if mid >= CULPRIT else RATE_BEFORE
        failures = sum(rng.random() < rate for _ in range(runs))
        tests += runs
        if failures >= threshold:
            bad = mid
        else:
            good = mid
    return bad, tests


def best_threshold(runs: int) -> int:
    """The failure count that best separates the two rates, from the binomial likelihoods."""
    def p(k, rate):
        return math.comb(runs, k) * rate ** k * (1 - rate) ** (runs - k)
    return next((k for k in range(runs + 1) if p(k, RATE_AFTER) > p(k, RATE_BEFORE)), runs + 1)


rng = random.Random(24)
print(f"{COMMITS} commits, the bug arrived in commit {CULPRIT}: failure rate {RATE_BEFORE:.0%} before, {RATE_AFTER:.0%} after\n")
print(f"  {'runs per step':>13}{'marked bad at':>15}{'found the culprit':>19}{'model runs, total':>19}")
for runs in (1, 3, 5, 10, 20, 40):
    t = best_threshold(runs)
    results = [bisect(runs, t, rng) for _ in range(TRIALS)]
    found = sum(commit == CULPRIT for commit, _ in results) / TRIALS
    print(f"  {runs:>13}{f'{t}+ failures':>15}{found:>19.0%}{sum(n for _, n in results) / TRIALS:>19.0f}")

steps = math.ceil(math.log2(COMMITS))
print(f"\neach step's verdict is a draw weighted by the two rates, and bisection trusts all {steps} of them:")
print(f"the search is right only when every step is")
