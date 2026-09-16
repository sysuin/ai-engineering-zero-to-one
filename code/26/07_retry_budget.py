# Retries during an outage, when failed attempts also add load. Three policies: no retries,
# up to three retries per request, and a retry budget of 20% of recent requests.
# The question is not only how many requests succeed, but whether the service recovers when
# the provider does.

import heapq
import random

RATE, CAPACITY = 80.0, 100.0          # requests a second offered; attempts a second the provider can serve
END, STEP = 120.0, 0.1
BUDGET, BURST = 0.2, 40.0            # retries may be 20% of requests, with a reserve of 40


def simulate(policy: str, OUTAGE: tuple, HEALTH_DURING: float, seed: int = 26) -> dict:
    rng = random.Random(seed)
    queue: list[tuple[float, int, int]] = []           # (due time, request id, attempt number)
    counter = 0
    tokens = 0.0                                       # the retry budget's bucket
    answered, requests, attempts_log = set(), 0, []
    success_log = []
    load = RATE                                        # attempts a second, smoothed over about a second
    t = 0.0
    while t < END:
        # fresh arrivals in this step
        for _ in range(sum(1 for _ in range(int(RATE * STEP * 3)) if rng.random() < 1 / 3)):
            heapq.heappush(queue, (t, counter, 0))
            counter += 1
            requests += 1
            tokens = min(tokens + BUDGET, BURST)        # each new request earns a fifth of a retry
        due = []
        while queue and queue[0][0] <= t:
            due.append(heapq.heappop(queue))
        load += (len(due) / STEP - load) * STEP          # a one-second moving average of the attempt rate
        healthy = 1.0 if not (OUTAGE[0] <= t < OUTAGE[1]) else HEALTH_DURING
        p = healthy * min(1.0, CAPACITY / load)
        ok = 0
        for _, rid, attempt in due:
            if rng.random() < p:
                answered.add(rid)
                ok += 1
                continue
            may_retry = (policy == "three retries" and attempt < 3) or \
                        (policy == "retry budget" and attempt < 3 and tokens >= 1.0)
            if may_retry:
                if policy == "retry budget":
                    tokens -= 1.0
                backoff = min(8.0, 0.5 * 2 ** attempt)
                heapq.heappush(queue, (t + rng.uniform(0, backoff), rid, attempt + 1))
        attempts_log.append((t, len(due)))
        success_log.append((t, ok, len(due)))
        t = round(t + STEP, 6)

    peak = max(sum(v for time, v in attempts_log if s <= time < s + 1) for s in range(int(END)))

    def success_rate(second):
        ok = sum(o for time, o, _ in success_log if second <= time < second + 1)
        tried = sum(n for time, _, n in success_log if second <= time < second + 1)
        return ok / tried if tried else 1.0

    recovered = next((s for s in range(int(OUTAGE[1]), int(END) - 5)
                      if all(success_rate(s + k) >= 0.95 for k in range(5))), None)
    return {"peak": peak, "answered": len(answered) / requests,
            "attempts": sum(v for _, v in attempts_log),
            "recovery": None if recovered is None else recovered - OUTAGE[1]}


SCENARIOS = [("a fifteen-second partial outage", (30.0, 45.0), 0.25),
             ("a one-second blip", (30.0, 31.0), 0.0)]
print(f"{RATE:.0f} requests a second offered to a provider that serves {CAPACITY:.0f} attempts a second")
for title, outage, health in SCENARIOS:
    print(f"\n{title}: from {outage[0]:.0f}s to {outage[1]:.0f}s it serves {health:.0%} of attempts\n")
    print(f"  {'policy':16}{'peak attempts/s':>16}{'upstream attempts':>19}{'answered':>10}{'healthy again after':>21}")
    for policy in ("no retries", "three retries", "retry budget"):
        r = simulate(policy, outage, health)
        back = "never, in the run" if r["recovery"] is None else f"{r['recovery']:.0f}s"
        print(f"  {policy:16}{r['peak']:>16,}{r['attempts']:>19,}{r['answered']:>10.1%}{back:>21}")
print("\n'healthy again' = seconds after the provider recovers until five straight seconds of 95%+ successful attempts")
