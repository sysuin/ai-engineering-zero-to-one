# timeout: 300
# Reads code/23/_latency.json. Logging everything is not affordable; how much
# fidelity does sampling actually cost?

import json
import random
import statistics

random.seed(23)
TRIALS = 2000
data = json.load(open("code/23/_latency.json"))
latency = data["latency"]
truth = {"p50": data["p50"], "p95": data["p95"], "p99": data["p99"]}


def pct(values, p):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(p / 100 * (len(values) - 1))))]


def sampled(rate: float, statistic: str) -> tuple[float, float]:
    """What a dashboard would report if it kept only `rate` of the traces."""
    keep = max(1, int(len(latency) * rate))
    estimates = sorted(pct(random.sample(latency, keep), int(statistic[1:]))
                       for _ in range(TRIALS))
    return estimates[int(0.05 * TRIALS)], estimates[int(0.95 * TRIALS)]


print(f"Measured on {len(latency)} requests. The question is what a sampled "
      f"dashboard sees.\n")
header = "  ".join(f"{name:>22}" for name in ("p50", "p95", "p99"))
truths = "  ".join(f"{'truth ' + format(truth[name], '.0f') + 'ms':>22}"
                   for name in ("p50", "p95", "p99"))
print(f"  {'kept':>6}  {header}")
print(f"  {'':>6}  {truths}")
rows = {}
for rate in (1.0, 0.5, 0.25, 0.1, 0.05):
    cells = []
    rows[rate] = {}
    for statistic in ("p50", "p95", "p99"):
        low, high = sampled(rate, statistic)
        rows[rate][statistic] = [low, high]
        cells.append(f"{low:.0f}-{high:.0f}ms")
    print(f"  {rate:>5.0%}  " + "  ".join(f"{c:>22}" for c in cells))

json.dump({"truth": truth, "rows": {str(k): v for k, v in rows.items()},
           "n": len(latency)}, open("code/23/_sampling.json", "w"), indent=1)

p50_10 = rows[0.1]["p50"]
p99_10 = rows[0.1]["p99"]
print()
print(f"At one trace in ten, the median is still readable — "
      f"{p50_10[0]:.0f} to {p50_10[1]:.0f}ms against a")
print(f"true {truth['p50']:.0f} — and p99 has become a wide guess "
      f"({p99_10[0]:.0f} to {p99_10[1]:.0f}ms).")
print()
print("That asymmetry is the whole of sampling policy. A median is a statement about")
print("the middle of the distribution and survives thinning; a p99 is a statement")
print("about the rarest one in a hundred, and thinning is exactly the operation that")
print("removes it.")
print()
print("So do not sample uniformly. Sample the way the questions differ:")
print()
print("  keep everything that failed        errors are rare and are the ones you")
print("                                     will be asked about")
print("  keep everything that was slow      a tail you have thrown away cannot be")
print("                                     investigated, only counted")
print("  keep everything a user marked      feedback without its trace is an opinion")
print("  sample the boring middle hard      1 in 100 is plenty to hold a p50 steady")
print()
print("This is tail-based sampling, and the name is the mechanism: the decision is")
print("made when the trace finishes and its duration and status are known, not when")
print("it starts. Head-based sampling — decide at the first span — is cheaper and")
print("throws away precisely the traces you wanted.")
