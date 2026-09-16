# An SLO on answer quality, and alerts on how fast its error budget is burning. Thirty days
# of simulated hourly samples, with a sharp incident and a slow decline in them.

import numpy as np

rng = np.random.default_rng(7)
SLO = 0.95                     # 95% of sampled answers judged correct, over 30 days
BUDGET = 1 - SLO               # so 5% may be wrong
PER_HOUR = 20                  # answers sampled and judged each hour
HOURS = 30 * 24

true_error = np.full(HOURS, 0.03)                  # healthy: 3% wrong
true_error[10 * 24 + 9: 10 * 24 + 15] = 0.40       # day 10, six bad hours
true_error[20 * 24:] = 0.09                        # from day 20, a slow decline
wrong = rng.binomial(PER_HOUR, true_error)

RULES = [("page", 2, 6.0), ("page", 6, 6.0), ("ticket", 72, 1.5)]   # (name, window, burn)


def burn(hour: int, window: int) -> float:
    start = max(0, hour - window + 1)
    errors = wrong[start: hour + 1].sum()
    return (errors / (PER_HOUR * (hour + 1 - start))) / BUDGET


print(f"SLO {SLO:.0%} correct over 30 days; {PER_HOUR} answers judged per hour\n")
for name, window, threshold in RULES:
    fired = [h for h in range(window - 1, HOURS) if burn(h, window) >= threshold]
    print(f"  {name:6} when the last {window:>2}h burn the budget at {threshold}x or faster "
          f"(error rate {threshold * BUDGET:.1%}+)")
    for label, start, end in (("the six-hour incident", 10 * 24 + 9, 10 * 24 + 15),
                              ("the slow decline", 20 * 24, HOURS)):
        after = [h for h in fired if start <= h < end + window]
        delay = f"after {after[0] - start + 1}h" if after else "never"
        print(f"         {label:22} detected {delay}")
    quiet = [h for h in fired if h < 10 * 24 + 9 or 10 * 24 + 15 + window <= h < 20 * 24]
    print(f"         hours alerting while healthy: {len(quiet)}")

spent = wrong.sum() / (PER_HOUR * HOURS)
print(f"\nover the month: {spent:.1%} of sampled answers wrong, "
      f"{spent / BUDGET:.0%} of the error budget spent")
