# Before you promise "at least 90%", work out how many labelled examples it takes to show it.

import math


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


TARGET, OBSERVED = 0.90, 0.93
print(f"You measure {OBSERVED:.0%} and the brief says 'at least {TARGET:.0%}'.\n")
print(f"  {'examples':>8} {'95% interval':>17} {'lower bound clears the target?':>31}")
needed = None
for n in (30, 60, 120, 250, 500, 1_000):
    lo, hi = wilson(round(OBSERVED * n), n)
    clears = lo > TARGET
    if clears and needed is None:
        needed = n
    print(f"  {n:>8,} {lo:>8.1%}–{hi:<7.1%} {('yes' if clears else 'no'):>31}")

print(f"\nwith {OBSERVED:.0%} observed, the first size in the table that demonstrates "
      f"'at least {TARGET:.0%}' is {needed:,}")
