# timeout: 300
# Availability and latency do not add the way a diagram suggests. Five compositions,
# computed, and simulated wherever the formula leans on an assumption.

import json

import numpy as np

rng = np.random.default_rng(31)
MINUTES = 30 * 24 * 60


def downtime(availability: float) -> str:
    return f"{(1 - availability) * MINUTES:,.0f} min/month"


# ------------------------------------------------------------------ 1. in series
# Illustrative monthly availabilities, not anyone's published figures.
chain = {"your service": 0.9995, "your database": 0.9995,
         "the embedding endpoint": 0.999, "the model endpoint": 0.999}
print("1. In series: every dependency must be up\n")
total = 1.0
for name, a in chain.items():
    total *= a
    print(f"  {name:<34} {a:>8.2%}   {downtime(a):>16}")
print(f"  {'the request, failures independent':<34} {total:>8.2%}   "
      f"{downtime(total):>16}")
# The embedding and model endpoints usually belong to one provider and fail together,
# which in series *helps*: two dependencies that are down at the same moments cost once.
shared = chain["your service"] * chain["your database"] * chain["the model endpoint"]
print(f"  {'the request, provider fails as one':<34} {shared:>8.2%}   "
      f"{downtime(shared):>16}")

# ------------------------------------------------------------------ 2. in parallel
print("\n2. A fallback provider: only as independent as their failures are\n")
u = 1 - 0.999
print(f"  {'share of outages in common':<28} {'available':>10} {'downtime':>16}")
fallback_rows = []
for shared in (0.0, 0.1, 0.5):
    # Common cause: your egress, your gateway, a region both depend on.
    unavailable = shared * u + ((1 - shared) * u) ** 2
    fallback_rows.append({"shared": shared, "available": 1 - unavailable})
    print(f"  {shared:<28.0%} {1 - unavailable:>10.4%} {downtime(1 - unavailable):>16}")

# ------------------------------------------------------------------ 3. latency in series
print("\n3. Latency in series: the p95 of a sum is not the sum of the p95s\n")
# Step times drawn independently. Real ones are partly correlated — a loaded provider slows
# every call in a request — which moves the two numbers below closer together.
N = 200_000
steps = {"embed the question": (0.25, 0.4), "retrieve": (0.05, 0.3),
         "model call 1": (1.6, 0.5), "model call 2": (1.6, 0.5),
         "model call 3": (1.6, 0.5)}
samples = {name: rng.lognormal(np.log(median), sigma, N)
           for name, (median, sigma) in steps.items()}
for name, s in samples.items():
    print(f"  {name:<22} p50 {np.percentile(s, 50):>5.2f}s   "
          f"p95 {np.percentile(s, 95):>5.2f}s")
whole = sum(samples.values())
sum_of_p95 = sum(np.percentile(s, 95) for s in samples.values())
p95_of_sum = np.percentile(whole, 95)
print(f"  {'sum of the p95s':<22} {sum_of_p95:>17.2f}s")
print(f"  {'p95 of the request':<22} {p95_of_sum:>17.2f}s")
print(f"\n  Budgeting each step at its own p95 over-states the request's p95 by "
      f"{sum_of_p95 / p95_of_sum - 1:.0%}: the")
print("  slow steps rarely coincide. A budget is set on the whole, and measured there.")

# ------------------------------------------------------------------ 4. fan-out
print("\n4. Fan-out: a request that waits for every one of n parallel calls\n")
SLOW = 0.01
print(f"  each call slow {SLOW:.0%} of the time")
fan_rows = []
for n in (1, 5, 10, 20, 50):
    formula = 1 - (1 - SLOW) ** n
    simulated = (rng.random((50_000, n)) < SLOW).any(axis=1).mean()
    fan_rows.append({"n": n, "slow": formula})
    print(f"  {n:>3} call{'s' if n > 1 else ' '}   request slow {formula:>5.1%}   "
          f"(simulated {simulated:.1%})")

# ------------------------------------------------------------------ 5. retries
print("\n5. Retries: fewer failures on a good day, more load on a bad one\n")
print(f"  {'call failure rate':<20} {'fails, 2 retries':>17} {'calls per request':>18}")
for p in (0.02, 0.2, 1.0):
    failed = p ** 3
    calls = 1 + p + p * p
    print(f"  {p:<20.0%} {failed:>17.4%} {calls:>18.2f}")
print("\n  At a 2% transient rate retries are nearly free. When the provider is down")
print("  they triple the traffic aimed at it, which is Chapter 26's breaker's job.")

json.dump({"chain": total, "fallback": fallback_rows, "sum_of_p95": sum_of_p95,
           "p95_of_sum": p95_of_sum, "fan_out": fan_rows},
          open("code/31/_composition.json", "w"), indent=1)
