# Some of every eval set's gold is wrong. What that does to a comparison between two systems
# depends on how it is wrong: at random, or in the way that punishes the newer system's answers.

import math

import numpy as np

rng = np.random.default_rng(13)
CASES, TRIALS = 120, 4_000
A_RIGHT, FIXES, BREAKS = 0.85, 0.08, 0.02     # B fixes 8% of cases and breaks 2%: truly 6 points better


def mcnemar(fixed: int, broke: int) -> float:
    n = fixed + broke
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(fixed, broke) + 1)) / 2 ** n)


def trial(error: float, kind: str) -> tuple[float, bool]:
    a = rng.random(CASES) < A_RIGHT
    b = a.copy()
    u = rng.random(CASES)
    b[~a & (u < FIXES / (1 - A_RIGHT))] = True             # B fixes some of A's failures
    b[a & (u < BREAKS / A_RIGHT)] = False                  # and breaks a few of its successes
    wrong = rng.random(CASES) < error
    if kind == "random":
        # a wrong gold answer marks every reply to that case wrong
        a_seen, b_seen = a & ~wrong, b & ~wrong
    else:
        # the gold encodes A's way of answering: where B is right and A
        # is not, a wrong gold marks B wrong and A right
        differ = b & ~a
        a_seen = np.where(wrong & differ, True, a)
        b_seen = np.where(wrong & differ, False, b)
    fixed, broke = int(np.sum(b_seen & ~a_seen)), int(np.sum(a_seen & ~b_seen))
    return b_seen.mean() - a_seen.mean(), fixed > broke and mcnemar(fixed, broke) < 0.05


print(f"{CASES} cases; system B is truly {(FIXES - BREAKS) * 100:.0f} points better than A, "
      f"which is {A_RIGHT:.0%} right;\n{TRIALS:,} trials\n")
print(f"  {'wrong gold':>10}  {'how it is wrong':34}{'measured gain':>14}{'B found better':>16}")
print(f"  {'':>10}  {'':34}{'(points)':>14}{'(p < 0.05)':>16}")
for error in (0.0, 0.05, 0.10, 0.20):
    for kind, label in (("random", "at random"), ("encodes A", "encodes the old system's answers")):
        if error == 0 and kind != "random":
            continue
        runs = [trial(error, kind) for _ in range(TRIALS)]
        gain = np.mean([g for g, _ in runs])
        found = np.mean([f for _, f in runs])
        print(f"  {error:>10.0%}  {label if error else 'none':34}{gain * 100:>+14.1f}{found:>16.0%}")
print("\nmeasured gain: the mean difference between the two scores")
print("B found better: the share of trials in which an exact McNemar test finds B better")
