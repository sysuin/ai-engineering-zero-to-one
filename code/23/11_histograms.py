# What a metrics system stores instead of every latency: buckets. How much a percentile read
# back from buckets depends on where the edges are, and why buckets, unlike percentiles, add up.

import math

import numpy as np

rng = np.random.default_rng(23)


def latencies(n: int, median_ms: float = 3_700, spread: float = 0.45) -> np.ndarray:
    return rng.lognormal(np.log(median_ms), spread, n)          # the same shape as 06


def from_fixed(counts: np.ndarray, edges: list[float], q: float) -> float:
    """Read a percentile back from bucket counts the way Prometheus's histogram_quantile does:
    find the bucket the rank falls in and interpolate linearly inside it."""
    rank, seen = q / 100 * counts.sum(), 0
    for i, count in enumerate(counts):
        if seen + count >= rank:
            low = edges[i - 1] if i else 0.0
            high = edges[i] if i < len(edges) else edges[-1]    # the overflow bucket has no top
            return low + (high - low) * (rank - seen) / count
        seen += count
    return edges[-1]


def fixed_counts(values: np.ndarray, edges: list[float]) -> np.ndarray:
    return np.bincount(np.searchsorted(edges, values), minlength=len(edges) + 1)


class LogSketch:
    """Edges grow by a constant factor, so every bucket has the same
    relative width and a value read back is within a fraction alpha of
    every value in its bucket. The idea behind DDSketch, and behind
    the exponential histograms of OpenTelemetry and Prometheus."""

    def __init__(self, alpha: float):
        self.gamma = (1 + alpha) / (1 - alpha)
        self.counts: dict[int, int] = {}

    def add(self, values: np.ndarray) -> "LogSketch":
        i = np.ceil(np.log(values) / math.log(self.gamma)).astype(int)
        idx, n = np.unique(i, return_counts=True)
        for i, c in zip(idx.tolist(), n.tolist()):
            self.counts[i] = self.counts.get(i, 0) + c
        return self

    def merge(self, other: "LogSketch") -> "LogSketch":
        for i, c in other.counts.items():
            self.counts[i] = self.counts.get(i, 0) + c
        return self

    def quantile(self, q: float) -> float:
        rank, seen = q / 100 * sum(self.counts.values()), 0
        for i in sorted(self.counts):
            seen += self.counts[i]
            if seen >= rank:
                return 2 * self.gamma ** i / (self.gamma + 1)
        raise ValueError("empty")


DEFAULT_S = [.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10]  # the Go client's default, seconds
DEFAULT = [1000 * e for e in DEFAULT_S]
EVEN = [1000.0 * s for s in range(1, 21)]                        # one second wide, up to 20s

sample = latencies(100_000)
truth = {q: np.percentile(sample, q) for q in (50, 95, 99)}
print("100,000 latencies; each estimate's error against the exact percentile\n")
print(f"  {'stored as':28}{'buckets':>8}{'p50':>11}{'p95':>11}{'p99':>11}")
print(f"  {'every value (exact)':28}{'-':>8}" + "".join(f"{truth[q]:>9,.0f}ms" for q in truth))


def show(name, buckets, estimates):
    cells = "".join(f"{e - truth[q]:>+9,.0f}ms" for q, e in zip(truth, estimates))
    print(f"  {name:28}{buckets:>8}{cells}")


for name, edges in (("default edges, 5ms to 10s", DEFAULT), ("even 1s edges, 1s to 20s", EVEN)):
    counts = fixed_counts(sample, edges)
    show(name, len(edges) + 1, [from_fixed(counts, edges, q) for q in truth])
for alpha in (0.05, 0.01):
    sketch = LogSketch(alpha).add(sample)
    show(f"log buckets, alpha {alpha:.0%}", len(sketch.counts), [sketch.quantile(q) for q in truth])
if truth[99] > DEFAULT[-1]:
    print(f"  (the default edges stop at {DEFAULT_S[-1]:g}s: the p99 fell in the overflow bucket,\n"
          "   which can only say 'more than that', and the top edge is what comes back)")

worst = max(abs(LogSketch(0.01).add(sample).quantile(q) - np.percentile(sample, q))
            / np.percentile(sample, q) for q in np.arange(1, 100))
print(f"\nlog buckets at alpha 1%, p1 to p99: worst relative error {worst:.2%}")
span = math.log(100_000 / 1) / math.log(1.01 / 0.99)
print(f"1ms to 100s at alpha 1% needs at most {math.ceil(span):,} buckets, "
      "whatever the traffic")

# The day from 06: quiet slow nights, busy fast days. Hourly percentiles cannot be combined;
# hourly buckets can, by adding the counts.
hours = [(200 if 8 <= h < 20 else 20, 3_000 if 8 <= h < 20 else 6_000) for h in range(24)]
day = [latencies(count, median) for count, median in hours]
exact = np.percentile(np.concatenate(day), 95)
merged = LogSketch(0.01)
for h in day:
    merged.merge(LogSketch(0.01).add(h))
print(f"\na day of {sum(len(h) for h in day):,} requests, kept as 24 hourly records")
print(f"  p95 of every request                {exact:>7,.0f}ms")
print(f"  mean of the hourly p95s             {np.mean([np.percentile(h, 95) for h in day]):>7,.0f}ms")
print(f"  p95 of the 24 sketches added up     {merged.quantile(95):>7,.0f}ms   "
      f"({merged.quantile(95) / exact - 1:+.1%})")
