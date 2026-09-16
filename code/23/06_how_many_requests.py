# Two percentile mistakes, measured on synthetic latencies with a known answer: estimating
# a p99 from too few requests, and averaging hourly percentiles into a daily one.

import numpy as np

rng = np.random.default_rng(23)


def latencies(n: int, median_ms: float = 3_700, spread: float = 0.45) -> np.ndarray:
    """Log-normal: most requests near the median, a long right tail — like agent runs."""
    return rng.lognormal(np.log(median_ms), spread, n)


truth = {q: np.percentile(latencies(2_000_000), q) for q in (50, 95, 99)}
print("true p50 {:,.0f}ms, p95 {:,.0f}ms, p99 {:,.0f}ms\n".format(*truth.values()))

print("  requests   95% of estimates of p50        of p95              of p99")
for n in (40, 400, 4_000, 40_000):
    cells = []
    for q in (50, 95, 99):
        estimates = [np.percentile(latencies(n), q) for _ in range(400)]
        low, high = np.percentile(estimates, [2.5, 97.5])
        cells.append(f"{low:>6,.0f}-{high:<6,.0f}")
    print(f"  {n:>8,}   " + "   ".join(cells))

# A day of traffic: quiet slow hours overnight, busy fast hours in the day.
hours = [(200 if 8 <= h < 20 else 20, 3_000 if 8 <= h < 20 else 6_000) for h in range(24)]
day = [latencies(count, median) for count, median in hours]
hourly_p95 = [np.percentile(h, 95) for h in day]
print(f"\na day of traffic, {sum(c for c, _ in hours):,} requests: busy fast daytime hours, "
      "quiet slow nights")
print(f"  p95 of the whole day           {np.percentile(np.concatenate(day), 95):>7,.0f}ms")
print(f"  mean of the 24 hourly p95s     {np.mean(hourly_p95):>7,.0f}ms   <- a number with no definition")
print(f"  median of the hourly p95s      {np.median(hourly_p95):>7,.0f}ms   <- still not a p95")
