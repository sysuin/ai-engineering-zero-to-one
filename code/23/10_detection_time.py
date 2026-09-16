# How long does it take to notice that answer quality fell, when quality is estimated from a
# sample that somebody or something grades? The judged sample rate is the dial.

import math

import numpy as np

rng = np.random.default_rng(23)
BASELINE, DEGRADED, TRIALS, WEEK = 0.95, 0.85, 400, 24 * 7
ALPHA = 0.001                  # one-sided: the test runs every hour, all week
ENOUGH = 100                   # judged answers per window, so the test has power


def critical(n: int) -> int:
    """The largest count of correct answers that is below 95% at the 0.1% level, exactly."""
    total, k = 0.0, -1
    while True:
        p = math.comb(n, k + 1) * BASELINE ** (k + 1) * (1 - BASELINE) ** (n - k - 1)
        if total + p > ALPHA:
            return k
        total, k = total + p, k + 1


def alarms(correct: np.ndarray, per_hour: int, window: int) -> np.ndarray:
    c = np.convolve(correct, np.ones(window), "valid")
    return c <= critical(per_hour * window)


print(f"quality falls from {BASELINE:.0%} to {DEGRADED:.0%}; every hour the last window of "
      "judged answers is tested\n")
print(f"  {'judged/hour':>11} {'window':>8} {'hours to notice':>16} {'false alarms a week':>20}")
rows = []
for per_hour in (5, 20, 50, 200):
    window = math.ceil(ENOUGH / per_hour)
    detect, false = [], []
    for _ in range(TRIALS):
        healthy = rng.binomial(per_hour, BASELINE, WEEK)
        false.append(int(alarms(healthy, per_hour, window).any()))
        # the fall happens at hour zero; the window still holds healthy hours before it
        history = np.concatenate([rng.binomial(per_hour, BASELINE, window),
                                  rng.binomial(per_hour, DEGRADED, 72)])
        fired = np.flatnonzero(alarms(history, per_hour, window))
        detect.append(fired[0] + 1 if len(fired) else math.inf)
    median = float(np.median(detect))
    rows.append((per_hour, window, median))
    shown = "over 3 days" if math.isinf(median) else f"{median:.0f}"
    print(f"  {per_hour:>11} {window:>7}h {shown:>16} {np.mean(false):>19.0%}")

slow, fast = rows[0], rows[-1]
print(f"\nAt {slow[0]} judged answers an hour the test needs a {slow[1]}-hour window to hold "
      "enough of them,")
print(f"so a fall takes about {slow[2]:.0f} hours to show; at {fast[0]} an hour it takes about "
      f"{fast[2]:.0f}.")
print("The false-alarm column is the share of healthy weeks with any alarm at all. Judge")
print("calls cost money, and the hours between a fall and a page cost wrong answers:")
print("choose the rate from the second, not the first.")
