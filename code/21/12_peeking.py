# Eval-driven development tempts you to look at a comparison as the cases come in and stop when
# it looks significant. Two systems, a paired test after every twenty cases: how often does
# peeking declare a winner that does not exist, and what does a threshold built for peeking cost?

import math

import numpy as np

rng = np.random.default_rng(2021)
TRIALS, MAX_CASES, LOOK_EVERY, ALPHA = 4_000, 400, 20, 0.05
LOOKS = MAX_CASES // LOOK_EVERY


def mcnemar(fixed: int, broke: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts."""
    n = fixed + broke
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(fixed, broke) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def trials(gain: float) -> np.ndarray:
    """p-value at every look, for every trial; 1 where B is not ahead. Shape trials x looks."""
    out = np.ones((TRIALS, LOOKS))
    for t in range(TRIALS):
        difficulty = rng.beta(4, 0.7, MAX_CASES)            # most cases easy, a few hard
        a = rng.random(MAX_CASES) < difficulty
        b = rng.random(MAX_CASES) < np.minimum(1, difficulty + gain)
        fixed, broke = np.cumsum(b & ~a), np.cumsum(a & ~b)
        for look in range(LOOKS):
            n = (look + 1) * LOOK_EVERY - 1
            if fixed[n] > broke[n]:
                out[t, look] = mcnemar(int(fixed[n]), int(broke[n]))
    return out


def report(p: np.ndarray, label: str, threshold: float, peek: bool) -> None:
    if peek:
        hit = p < threshold
        declared = hit.any(axis=1)
        used = np.where(declared, (hit.argmax(axis=1) + 1) * LOOK_EVERY, MAX_CASES).mean()
    else:
        declared, used = p[:, -1] < threshold, MAX_CASES
    print(f"  {label:38}{declared.mean():>10.1%}{used:>11.0f}")


null, better = trials(0.0), trials(0.05)
single = (null[:, -1] < ALPHA).mean()
# Pocock's idea: one threshold at every look, set so that the chance of
# any false declaration across all looks equals the single test's.
pocock = float(np.quantile(null.min(axis=1), single))

print(f"up to {MAX_CASES} paired cases, a look after every {LOOK_EVERY}; {TRIALS:,} trials per row\n")
print(f"  {'rule':38}{'declares':>10}{'cases':>11}")
print(f"  {'':38}{'B better':>10}{'used':>11}")
for gain, p in (("no true difference", null), ("B truly 5 points better", better)):
    print(f"  {gain}")
    report(p, f"one test at the end, p < {ALPHA}", ALPHA, peek=False)
    report(p, f"peek, stop at p < {ALPHA}", ALPHA, peek=True)
    report(p, f"peek, stop at p < {ALPHA}/{LOOKS} (Bonferroni)", ALPHA / LOOKS, peek=True)
    report(p, f"peek, stop at p < {pocock:.4f} (Pocock)", pocock, peek=True)
print("\nwith no true difference, 'declares B better' is the false-positive rate;")
print("with one, it is the power")
