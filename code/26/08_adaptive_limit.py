# A bulkhead's size is usually a constant. The provider's capacity is not. A fixed concurrency
# limit against one that adapts — additive increase on success, a cut on a 429 — while the
# provider's capacity drops to a third and comes back.

import random

STEP, END, WORK = 0.05, 120.0, 2.0     # seconds per step, run length, seconds per model call


def capacity(t: float) -> int:
    return 6 if 40.0 <= t < 80.0 else 18


def simulate(policy: str, fixed: int = 18, backoff: float = 0.5, seed: int = 26) -> dict:
    rng = random.Random(seed)
    limit, running, t = float(fixed), [], 0.0
    done = rejected = 0
    limits, done_in = [], {"before": 0, "during": 0, "after": 0}
    while t < END:
        running = [end for end in running if end > t]
        # start as many calls as the limit allows; demand is always there
        while len(running) < int(limit):
            if len(running) >= capacity(t):             # the provider refuses: an immediate 429
                rejected += 1
                if policy == "adaptive":
                    limit = max(1.0, limit * backoff)
                else:
                    break
                continue
            running.append(t + WORK * rng.uniform(0.8, 1.2))
            if policy == "adaptive":
                limit = min(64.0, limit + 1 / max(limit, 1.0))
        for end in running:
            if t < end <= t + STEP:
                done += 1
                phase = "before" if t < 40 else "during" if t < 80 else "after"
                done_in[phase] += 1
        limits.append((t, limit))
        t = round(t + STEP, 6)
    during = [l for time, l in limits if 40 <= time < 80]
    return {"done": done, "rejected": rejected, "per_s": {k: v / 40 for k, v in done_in.items()},
            "limit_during": sum(during) / len(during)}


print("demand never stops; each call takes about 2s; the provider serves 18 at once, 6 between 40s and 80s\n")
print(f"  {'client limit':22}{'429s':>7}{'answered/s before':>19}{'during':>8}{'after':>7}{'limit, during':>15}")
for label, policy, fixed, backoff in (("fixed at 18", "fixed", 18, 0), ("fixed at 6", "fixed", 6, 0),
                                     ("adaptive, halve on 429", "adaptive", 18, 0.5),
                                     ("adaptive, x0.9 on 429", "adaptive", 18, 0.9)):
    r = simulate(policy, fixed, backoff)
    print(f"  {label:22}{r['rejected']:>7,}{r['per_s']['before']:>19.1f}{r['per_s']['during']:>8.1f}"
          f"{r['per_s']['after']:>7.1f}{r['limit_during']:>15.1f}")
