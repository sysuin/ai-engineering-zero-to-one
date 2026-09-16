# Past capacity, a server can stay busy and still deliver almost nothing: it finishes work
# for callers who have already given up. Checking the caller's deadline before starting the
# work is the difference. A discrete-event simulation, so the numbers are exact and free.

import heapq

import numpy as np

rng = np.random.default_rng(26)
WORKERS, SERVICE, CLIENT_TIMEOUT, SECONDS = 8, 2.0, 6.0, 600


def simulate(rate: float, check_deadline: bool) -> dict:
    arrivals = np.cumsum(rng.exponential(1 / rate, int(rate * SECONDS * 1.1)))
    arrivals = arrivals[arrivals < SECONDS]
    durations = rng.lognormal(np.log(SERVICE) - 0.18, 0.6, len(arrivals))
    free = [0.0] * WORKERS
    useful = wasted = dropped = 0
    for arrived, work in zip(arrivals, durations):
        start = max(arrived, heapq.heappop(free))
        deadline = arrived + CLIENT_TIMEOUT
        # Start only if the work can plausibly finish in time: remaining time must cover
        # the typical duration, not merely be positive.
        if check_deadline and deadline - start < SERVICE:
            heapq.heappush(free, start)           # dropped at the head of the queue: free
            dropped += 1
            continue
        finish = start + work
        heapq.heappush(free, finish)
        if finish <= deadline:
            useful += 1
        else:
            wasted += 1                           # finished for a caller who had left
    return {"useful": useful / SECONDS, "wasted": wasted / SECONDS, "dropped": dropped,
            "arrived": len(arrivals) / SECONDS}


capacity = WORKERS / SERVICE
print(f"{WORKERS} workers, {SERVICE:.0f}s of work on average: capacity {capacity:.0f} requests a "
      f"second. Callers give up after {CLIENT_TIMEOUT:.0f}s.\n")
print(f"  {'':>10}   {'no deadline check':<28} {'deadline checked first':<27}")
print(f"  {'arrivals/s':>10}   {'in time':>16} {'wasted':>9}   {'in time':>16} {'wasted':>9}")
rows = []
for load in (0.8, 1.0, 1.2, 1.5, 2.0):
    rate = load * capacity
    plain, checked = simulate(rate, False), simulate(rate, True)
    rows.append((load, plain, checked))
    print(f"  {rate:>10.1f}   {plain['useful']:>15.2f}/s {plain['wasted']:>8.2f}/s"
          f"   {checked['useful']:>15.2f}/s {checked['wasted']:>8.2f}/s")

load, plain, checked = rows[-1]
print(f"\nAt {load:.0f}x capacity the unchecked server stayed fully busy and answered "
      f"{plain['useful']:.2f} requests a")
print(f"second in time. Starting a request only when its caller would still be waiting after")
print(f"typical work answered {checked['useful']:.2f}, from the same workers.")
print("Throughput is what the server did. Goodput is what callers received, and past capacity")
print("the two part company unless the server stops starting work nobody is waiting for.")
