# One tenant's batch job and another tenant's analysts, sharing eight workers. The same
# arrivals under three ways of choosing the next request. Simulated: every rate below is an
# assumption stated here, and the point is how the choice of queue moves the wait.

import heapq
import json
import random
import statistics

WORKERS = 8
HORIZON = 900.0                     # seconds simulated
ANALYST_RATE = 4.0                  # tenant A: requests a second, around the clock
BATCH_AT, BATCH_SIZE = 100.0, 1_500  # tenant B: a batch job submitted all at once


def arrivals(rng: random.Random) -> list[tuple[float, str, float]]:
    """(arrival time, tenant, seconds of work), in time order."""
    out, t = [], 0.0
    while t < HORIZON:
        t += rng.expovariate(ANALYST_RATE)
        out.append((t, "A", rng.lognormvariate(-0.125, 0.5)))    # a mean of one second
    out += [(BATCH_AT, "B", rng.lognormvariate(-0.125, 0.5)) for _ in range(BATCH_SIZE)]
    return sorted(out)


def simulate(policy: str, jobs: list[tuple[float, str, float]], batch_cap: int):
    queues = {"A": [], "B": []}           # per tenant, in arrival order
    running = {"A": 0, "B": 0}
    events = [(t, 0, i) for i, (t, _, _) in enumerate(jobs)]   # kind 0 = arrival, 1 = done
    heapq.heapify(events)
    waits, batch_done, turn = [], 0.0, "A"
    while events:
        now, kind, i = heapq.heappop(events)
        tenant = jobs[i][1]
        if kind == 0:
            queues[tenant].append(i)
        else:
            running[tenant] -= 1
            if tenant == "B":
                batch_done = max(batch_done, now)
        while sum(running.values()) < WORKERS:
            ready = [t for t in "AB" if queues[t]]
            oldest = lambda t: jobs[queues[t][0]][0]      # noqa: E731
            if policy == "one queue":
                # whoever arrived first, whatever the tenant
                pick = min(ready, key=oldest, default=None)
            elif policy == "take turns":
                # alternate between the tenants that are waiting
                pick = turn if turn in ready else (ready or [None])[0]
                turn = "B" if pick == "A" else "A"
            else:
                # the batch tenant may hold only a few workers
                allowed = [t for t in ready
                           if t == "A" or running["B"] < batch_cap]
                pick = min(allowed, key=oldest, default=None)
            if pick is None:
                break
            j = queues[pick].pop(0)
            running[pick] += 1
            if pick == "A" and BATCH_AT <= jobs[j][0] <= BATCH_AT + 300:
                waits.append(now - jobs[j][0])
            heapq.heappush(events, (now + jobs[j][2], 1, j))
    return waits, batch_done


jobs = arrivals(random.Random(26))
print(f"{WORKERS} workers; tenant A sends {ANALYST_RATE:.0f} requests a second; tenant B submits "
      f"{BATCH_SIZE:,} at t={BATCH_AT:.0f}s")
print("every request needs about a second of work\n")
print(f"  {'next request chosen by':<30}{'A waits, p50':>13}{'p95':>8}{'B finished at':>15}")
table = []
for policy, cap in (("one queue", 0), ("take turns", 0), ("batch capped", 3), ("batch capped", 2)):
    waits, done = simulate(policy, jobs, cap)
    p95 = statistics.quantiles(waits, n=20)[18]
    label = f"{policy} at {cap} workers" if cap else policy
    print(f"  {label:<30}{statistics.median(waits):>12.1f}s{p95:>7.1f}s{done:>14.0f}s")
    table.append({"policy": label, "p50": statistics.median(waits), "p95": p95,
                  "batch_minutes": (done - BATCH_AT) / 60})
json.dump(table, open("code/26/_noisy_neighbour.json", "w"), indent=1)
print("\n  A's waits are for requests sent in the five minutes after the batch arrived")
