# timeout: 300
# Little's law gives the average in flight. A design needs the busy moments, and the
# price of a pool that is too small. Both follow from arrivals that are random.

import heapq
import json
import math

import numpy as np

rng = np.random.default_rng(31)
envelope = json.load(open("code/31/_envelope.json"))["designs"]


def poisson_quantile(mean: float, q: float) -> int:
    """The smallest n with P(N <= n) >= q, for N ~ Poisson(mean)."""
    n, term = 0, math.exp(-mean)
    total = term
    while total < q:
        n += 1
        term *= mean / n
        total += term
    return n


def erlang_b(servers: int, load: float) -> float:
    """Chance an arrival finds every slot busy, when it is turned away rather than queued."""
    b = 1.0
    for k in range(1, servers + 1):
        b = load * b / (k + load * b)
    return b


def erlang_c(servers: int, load: float) -> float:
    """Chance an arrival has to wait, when it queues for a slot. Needs load < servers."""
    b = erlang_b(servers, load)
    return servers * b / (servers - load * (1 - b))


def durations(mean: float, n: int, sigma: float = 0.6) -> np.ndarray:
    """Lognormal times with the given mean: a long right tail, like model calls."""
    return rng.lognormal(math.log(mean) - sigma ** 2 / 2, sigma, n)


def arrivals(rate: float, seconds: float) -> np.ndarray:
    return np.cumsum(rng.exponential(1 / rate, int(rate * seconds * 1.2)))


# ------------------------------------------------------------------ 1. the busy moments
print("1. In flight: the average, and the moments a design has to survive\n")
print(f"  {'design':<44} {'mean':>6} {'p99':>5} {'p99.9':>6}")
for name, row in envelope.items():
    mean = row["concurrency"]
    print(f"  {name:<44} {mean:>6.1f} {poisson_quantile(mean, 0.99):>5} "
          f"{poisson_quantile(mean, 0.999):>6}")

name, row = max(envelope.items(), key=lambda kv: kv[1]["concurrency"])
rate = row["concurrency"] / row["latency"]
start = arrivals(rate, 4 * 3600)
finish = start + durations(row["latency"], len(start))
probes = np.sort(rng.uniform(3600, 3 * 3600, 20_000))       # skip the empty first hour
in_flight = (np.searchsorted(start, probes, side="right")
             - np.searchsorted(np.sort(finish), probes, side="right"))
print("\n  simulated for the busiest: two hours of random arrivals, lognormal durations")
print(f"    mean {in_flight.mean():.1f}   p99 {np.percentile(in_flight, 99):.0f}   "
      f"p99.9 {np.percentile(in_flight, 99.9):.0f}")

# ------------------------------------------------------------------ 2. a pool that refuses
SLOTS, SERVICE = 8, 10.0
print(f"\n2. A pool of {SLOTS} slots that turns work away when full, "
      f"like Clarity's bulkhead\n")
print(f"  {'offered load':>12} {'turned away':>12}   simulated")


def simulate_loss(load: float, seconds: float = 40 * 3600) -> float:
    times = arrivals(load / SERVICE, seconds)
    lengths = durations(SERVICE, len(times))
    busy: list[float] = []
    refused = 0
    for t, length in zip(times, lengths):
        while busy and busy[0] <= t:
            heapq.heappop(busy)
        if len(busy) >= SLOTS:
            refused += 1
        else:
            heapq.heappush(busy, t + length)
    return refused / len(times)


loss_rows = []
for load in (1, 2, 4, 6, 8):
    exact, sim = erlang_b(SLOTS, load), simulate_loss(load)
    loss_rows.append({"load": load, "erlang_b": exact, "simulated": sim})
    print(f"  {load:>12} {exact:>11.2%}   {sim:>8.2%}")

# ------------------------------------------------------------------ 3. a pool that queues
print(f"\n3. The same {SLOTS} slots, but arrivals wait for one: "
      f"a {SERVICE:.0f}s request's extra wait\n")
print(f"  {'utilisation':>11} {'must wait':>10} {'mean wait':>10} "
      f"{'lognormal sim':>14}")


def simulate_queue(load: float, seconds: float = 40 * 3600) -> float:
    times = arrivals(load / SERVICE, seconds)
    lengths = durations(SERVICE, len(times))
    free = [0.0] * SLOTS
    waited = 0.0
    for t, length in zip(times, lengths):
        soonest = heapq.heappop(free)
        begin = max(t, soonest)
        waited += begin - t
        heapq.heappush(free, begin + length)
    return waited / len(times)


queue_rows = []
for utilisation in (0.5, 0.7, 0.8, 0.9, 0.95):
    load = utilisation * SLOTS
    c = erlang_c(SLOTS, load)
    wait = c * SERVICE / (SLOTS - load)                      # exponential service times
    sim = simulate_queue(load)
    queue_rows.append({"utilisation": utilisation, "erlang_c": c, "wait": wait,
                       "simulated": sim})
    print(f"  {utilisation:>11.0%} {c:>10.0%} {wait:>9.1f}s {sim:>13.1f}s")

json.dump({"loss": loss_rows, "queue": queue_rows, "slots": SLOTS, "service": SERVICE},
          open("code/31/_in_flight.json", "w"), indent=1)

low, high = queue_rows[0], queue_rows[-1]
print(f"\nFrom {low['utilisation']:.0%} to {high['utilisation']:.0%} busy, the mean wait "
      f"grows {high['wait'] / low['wait']:.0f}-fold in the formula and "
      f"{high['simulated'] / low['simulated']:.0f}-fold")
print("in the simulation. The formula assumes exponential durations; the simulation's")
print("lognormal ones vary less and wait less, and bend at the same place.")
cv2 = math.exp(0.6 ** 2) - 1
ratios = [r["simulated"] / r["wait"] for r in queue_rows[1:]]
print(f"\nSimulated over formula, from {queue_rows[1]['utilisation']:.0%} busy up: "
      f"{min(ratios):.2f} to {max(ratios):.2f}.")
print(f"The Allen-Cunneen correction for durations this variable, (1 + CV^2) / 2,")
print(f"predicts {(1 + cv2) / 2:.2f}.")
