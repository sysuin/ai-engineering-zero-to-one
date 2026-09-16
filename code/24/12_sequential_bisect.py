# The bisect of 09_bisect_a_rate.py, with each commit tested until the runs have decided rather
# than for a fixed count. Wald's sequential probability ratio test, against the same two rates.

import json
import math
import random

COMMITS, CULPRIT = 64, 41
RATE_BEFORE, RATE_AFTER = 0.05, 0.30
TRIALS = 20_000


def fixed(runs: int, threshold: int):
    def verdict(rate: float, rng: random.Random) -> tuple[bool, int]:
        failures = sum(rng.random() < rate for _ in range(runs))
        return failures >= threshold, runs
    return verdict


def sequential(alpha: float, beta: float, most: int = 200):
    # The evidence one run adds: a failure points to "bad", a pass to "good".
    fail_step = math.log(RATE_AFTER / RATE_BEFORE)
    pass_step = math.log((1 - RATE_AFTER) / (1 - RATE_BEFORE))
    upper = math.log((1 - beta) / alpha)       # stop and call it bad
    lower = math.log(beta / (1 - alpha))       # stop and call it good

    def verdict(rate: float, rng: random.Random) -> tuple[bool, int]:
        evidence, runs = 0.0, 0
        while lower < evidence < upper and runs < most:
            evidence += fail_step if rng.random() < rate else pass_step
            runs += 1
        return evidence >= upper or (runs == most and evidence > 0), runs
    return verdict


def bisect(verdict, rng: random.Random) -> tuple[bool, int]:
    good, bad, total = 0, COMMITS, 0
    while bad - good > 1:
        mid = (good + bad) // 2
        is_bad, runs = verdict(RATE_AFTER if mid >= CULPRIT else RATE_BEFORE, rng)
        total += runs
        good, bad = (good, mid) if is_bad else (mid, bad)
    return bad == CULPRIT, total


rng = random.Random(24)
print(f"{COMMITS} commits, failure rate {RATE_BEFORE:.0%} before commit {CULPRIT} and "
      f"{RATE_AFTER:.0%} after; {TRIALS:,} bisects each\n")
print(f"  {'':<33}{'runs per commit':>17}")
print(f"  {'each commit is tested':<33}{'good':>7}{'bad':>10}{'found it':>10}{'runs in all':>13}")
table = []
for label, verdict in (("20 runs, bad at 3+ failures", fixed(20, 3)),
                       ("40 runs, bad at 6+ failures", fixed(40, 6)),
                       ("until decided, 5% error each way", sequential(0.05, 0.05)),
                       ("until decided, 1% error each way", sequential(0.01, 0.01))):
    good_runs = sum(verdict(RATE_BEFORE, rng)[1] for _ in range(2_000)) / 2_000
    bad_runs = sum(verdict(RATE_AFTER, rng)[1] for _ in range(2_000)) / 2_000
    results = [bisect(verdict, rng) for _ in range(TRIALS)]
    found = sum(r[0] for r in results) / TRIALS
    runs = sum(r[1] for r in results) / TRIALS
    print(f"  {label:<33}{good_runs:>7.1f}{bad_runs:>10.1f}{found:>10.0%}{runs:>13.0f}")
    table.append({"label": label, "found": found, "runs": runs})
json.dump(table, open("code/24/_sequential.json", "w"), indent=1)
