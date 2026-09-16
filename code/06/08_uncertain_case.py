# A tornado moves one input at a time. Real estimates are all uncertain at once. Draw every
# input from a range and look at the whole distribution of payback, not one number.

import random
import statistics

# (low, most likely, high) for each input: the base case of 04_sensitivity.py is the middle column
RANGES = {
    "tickets per year":       (10_000, 20_000, 30_000),
    "minutes per ticket":     (1.0, 1.75, 3.0),
    "hourly rate ($)":        (28.0, 34.0, 42.0),
    "share handled (0-1)":    (0.60, 0.80, 0.90),
    "build days":             (8, 12, 24),
    "errors per 100 handled": (2.0, 4.0, 10.0),
    "cost of an error ($)":   (3.0, 6.0, 15.0),
}
DAY_RATE, DRAWS = 520.0, 100_000


def payback_years(p: dict) -> float:
    handled = p["tickets per year"] * p["share handled (0-1)"]
    saved = handled * p["minutes per ticket"] / 60 * p["hourly rate ($)"]
    errors = handled * p["errors per 100 handled"] / 100 * p["cost of an error ($)"]
    net = saved - errors
    return float("inf") if net <= 0 else p["build days"] * DAY_RATE / net


rng = random.Random(6)
draws = []
for _ in range(DRAWS):
    p = {name: rng.triangular(low, high, mode) for name, (low, mode, high) in RANGES.items()}
    draws.append((p, payback_years(p)))
years = sorted(y for _, y in draws)

base = payback_years({name: mode for name, (_, mode, _) in RANGES.items()})
finite = [y for y in years if y != float("inf")]
print(f"base case, every input at its most likely value: payback {base:.2f} years\n")
print(f"{DRAWS:,} draws with every input uncertain at once:")
print(f"  median payback            {statistics.median(years):.2f} years")
print(f"  10th to 90th percentile   {years[DRAWS // 10]:.2f} to {years[9 * DRAWS // 10]:.2f} years")
print(f"  pays back within a year   {sum(y <= 1 for y in years) / DRAWS:.0%} of draws")
print(f"  never pays back           {sum(y == float('inf') for y in years) / DRAWS:.1%} of draws")


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0] * len(values)
    for rank, i in enumerate(order):
        out[i] = rank
    return out


capped = [min(y, 20.0) for _, y in draws]         # 'never' ranks as the longest payback
payback_rank = ranks(capped)
print("\nwhich uncertainty matters most (rank correlation with payback):")
scores = []
for name in RANGES:
    r = statistics.correlation(ranks([p[name] for p, _ in draws]), payback_rank)
    scores.append((abs(r), r, name))
for _, r, name in sorted(scores, reverse=True):
    print(f"  {name:24} {r:+.2f}")
