# Which assumption decides the answer? Move each input by the same proportion, one at a
# time, and see how far the payback moves. The widest bar is the number to go and measure.

BASE = {
    "tickets per year":       20_000,
    "minutes per ticket":     1.75,
    "hourly rate ($)":        34.0,
    "share handled (0-1)":    0.80,
    "build days":             12,
    "errors per 100 handled": 4.0,
    "cost of an error ($)":   6.0,
}
DAY_RATE = 520.0


def payback_years(p: dict) -> float:
    handled = p["tickets per year"] * p["share handled (0-1)"]
    saved = handled * p["minutes per ticket"] / 60 * p["hourly rate ($)"]
    errors = handled * p["errors per 100 handled"] / 100 * p["cost of an error ($)"]
    net = saved - errors
    return float("inf") if net <= 0 else p["build days"] * DAY_RATE / net


base = payback_years(BASE)
print(f"base case: payback {base:.2f} years\n")
print(f"  {'input moved by ±50%':24} {'low':>8} {'high':>8}   swing")

swings = []
for name, value in BASE.items():
    results = []
    for factor in (0.5, 1.5):
        moved = dict(BASE)
        moved[name] = min(value * factor, 1.0) if name.startswith("share") else value * factor
        results.append(payback_years(moved))
    lo, hi = min(results), max(results)
    swings.append((hi - lo, name, lo, hi))

for swing, name, lo, hi in sorted(swings, reverse=True):
    bar = "█" * min(40, round(swing * 40))
    print(f"  {name:24} {lo:>7.2f}y {hi:>7.2f}y   {bar}")

top = max(swings)[0]
widest = [name for swing, name, _, _ in swings if abs(swing - top) < 1e-9]
print(f"\nmoves the answer most: {' and '.join(widest)}"
      + (" — tied, because payback depends on their product" if len(widest) > 1 else ""))

# The same analysis at a worse error rate, where the error term stops being small.
worse = dict(BASE, **{"errors per 100 handled": 10.0})
print(f"\nat 10 errors per 100 handled, payback is {payback_years(worse):.2f} years, and:")
for name, factor in (("errors per 100 handled", 0.5), ("errors per 100 handled", 1.5),
                     ("minutes per ticket", 0.5)):
    moved = dict(worse, **{name: worse[name] * factor})
    years = payback_years(moved)
    print(f"  {name} x{factor}: " + ("never pays back" if years == float("inf") else f"{years:.2f} years"))
