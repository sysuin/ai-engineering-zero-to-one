# Two ways to notice that quality fell, on the same stream of judged answers and held to the
# same false-alarm rate: a test over a fixed window, and a cumulative sum that updates with
# every answer (Page's CUSUM).

import math

import numpy as np

rng = np.random.default_rng(2023)
BASELINE, DEGRADED, PER_HOUR, WEEK = 0.95, 0.85, 50, 24 * 7
WINDOW, ALPHA, TRIALS = 2, 0.001, 2_000              # the 50-an-hour row of 10_detection_time

# Each judged answer moves the sum by the log of how much likelier it
# is after a fall than before one. A correct answer is weak evidence
# of health; a wrong one is strong evidence of a fall.
RIGHT = math.log(DEGRADED / BASELINE)
WRONG = math.log((1 - DEGRADED) / (1 - BASELINE))


def critical(n: int) -> int:
    total, k = 0.0, -1
    while True:
        p = math.comb(n, k + 1) * BASELINE ** (k + 1) * (1 - BASELINE) ** (n - k - 1)
        if total + p > ALPHA:
            return k
        total, k = total + p, k + 1


def window_alarm_hour(judged: np.ndarray) -> np.ndarray:
    """judged: trials x answers, 1 for correct. Tested at the end of every hour."""
    hourly = judged.reshape(len(judged), -1, PER_HOUR).sum(axis=2)
    windowed = np.lib.stride_tricks.sliding_window_view(hourly, WINDOW, axis=1).sum(axis=2)
    fired = windowed <= critical(PER_HOUR * WINDOW)
    first = np.where(fired.any(axis=1), fired.argmax(axis=1) + WINDOW, np.inf)
    return first                                      # hours, counted from the first answer


def cusum_path(judged: np.ndarray) -> np.ndarray:
    """The running sum after every answer, floored at zero."""
    s = np.zeros(len(judged))
    path = np.empty(judged.shape)
    steps = np.where(judged == 1, RIGHT, WRONG)
    for t in range(judged.shape[1]):
        s = np.maximum(0.0, s + steps[:, t])
        path[:, t] = s
    return path


def answers(hours: int, quality: float) -> np.ndarray:
    return (rng.random((TRIALS, hours * PER_HOUR)) < quality).astype(int)


# Calibrate: pick the CUSUM threshold that gives the same share of healthy weeks with an alarm
# as the window test has. That keeps the comparison about speed, not about nerve.
healthy_weeks = answers(WEEK, BASELINE)
window_false = np.isfinite(window_alarm_hour(healthy_weeks)).mean()
calibration = cusum_path(healthy_weeks).max(axis=1)
h = float(np.quantile(calibration, 1 - window_false))
print(f"{PER_HOUR} judged answers an hour; a fall from {BASELINE:.0%} to {DEGRADED:.0%}")
print(f"a correct answer moves the sum by {RIGHT:+.3f}, a wrong one by {WRONG:+.3f}")
print(f"both rules held to {window_false:.0%} of healthy weeks with a false alarm; "
      f"CUSUM threshold {h:.2f}\n")

check = answers(WEEK, BASELINE)                       # fresh weeks, not the calibration ones
print(f"  false alarms on fresh healthy weeks: window {np.isfinite(window_alarm_hour(check)).mean():.0%}, "
      f"CUSUM {(cusum_path(check).max(axis=1) > h).mean():.0%}\n")

print(f"  {'quality after':<24}{'window test':>26}{'CUSUM':>26}")
print(f"  {'the fall':<24}" + f"{'median':>9}{'90th pct':>10}{'found':>7}" * 2)
WARM, AFTER = 24, 48                                  # a healthy day first, so neither starts fresh


def summary(times: np.ndarray) -> str:
    """Hours from the fall to the first alarm; runs that alarmed during the healthy day are left
    out, and runs with no alarm in two days count as not found."""
    times = np.sort(times[times > 0])
    found = np.isfinite(times).mean()
    med, p90 = times[len(times) // 2], times[int(0.9 * len(times))]
    fmt = lambda x: "  > 48" if math.isinf(x) else f"{x:>6.1f}"
    return f"{fmt(med):>9}{fmt(p90):>10}{found:>7.0%}"


for after in (DEGRADED, 0.90, 0.93):
    history = np.concatenate([answers(WARM, BASELINE), answers(AFTER, after)], axis=1)
    w = window_alarm_hour(history) - WARM
    fired = cusum_path(history) > h
    c = np.where(fired.any(axis=1), (fired.argmax(axis=1) + 1) / PER_HOUR, np.inf) - WARM
    print(f"  {after:>8.0%}{'':16}{summary(w)}{summary(c)}")
print("\n  hours from the fall to the first alarm; 'found' = alarmed within two days")
