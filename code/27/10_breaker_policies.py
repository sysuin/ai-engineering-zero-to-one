# With a fallback behind it, an open breaker does not cause an outage: it moves traffic to the
# second model. So the question a gateway's breaker answers is different from Chapter 26's —
# how many calls are wasted on a failing primary, and how much traffic is sent to the fallback
# while the primary was fine. Simulated across a healthy hour, a flaky hour and a dead one.

import numpy as np

rng = np.random.default_rng(2027)
PER_PHASE = 3_000
PHASES = [("healthy", 0.01), ("flaky", 0.25), ("down", 1.0), ("recovered", 0.01)]


class Consecutive:
    """Open after k failures in a row; after a cooldown, let one call
    through to probe."""
    def __init__(self, k, cooldown):
        self.k, self.cooldown = k, cooldown
        self.fails, self.opened = 0, None

    def allow(self, t):
        return self.opened is None or t - self.opened >= self.cooldown

    def record(self, t, ok):
        if ok:
            self.fails, self.opened = 0, None
        else:
            self.fails += 1
            if self.fails >= self.k:
                self.opened = t


class Rate:
    """Open when more than a share of the last n calls failed; probe after a cooldown."""
    def __init__(self, n, share, cooldown):
        self.n, self.share, self.cooldown, self.window, self.opened = n, share, cooldown, [], None

    def allow(self, t):
        return self.opened is None or t - self.opened >= self.cooldown

    def record(self, t, ok):
        if self.opened is not None and ok:
            self.opened, self.window = None, []
        self.window = (self.window + [ok])[-self.n:]
        if len(self.window) == self.n and self.window.count(False) / self.n > self.share:
            self.opened = t


class Never:
    def allow(self, t):
        return True

    def record(self, t, ok):
        pass


def simulate(breaker):
    rows = []
    t = 0
    for name, error in PHASES:
        wasted = fallback = 0
        for _ in range(PER_PHASE):
            t += 1
            if breaker.allow(t):
                ok = rng.random() > error
                breaker.record(t, ok)
                if not ok:
                    wasted += 1
                    fallback += 1            # a failed primary call, then the fallback
            else:
                fallback += 1                # sent straight to the fallback
        rows.append((name, wasted / PER_PHASE, fallback / PER_PHASE))
    return rows


POLICIES = [("no breaker", Never()),
            ("3 failures in a row, cool 50", Consecutive(3, 50)),
            ("10 failures in a row, cool 50", Consecutive(10, 50)),
            ("over 20% of last 20, cool 50", Rate(20, 0.2, 50)),
            ("over 50% of last 20, cool 50", Rate(20, 0.5, 50))]

print(f"{PER_PHASE:,} requests per phase; the primary's error rate in each:")
print("  " + ", ".join(f"{n} {e:.0%}" for n, e in PHASES))
print("a cooldown is counted in requests\n")
print(f"  {'breaker':31}" + "".join(f"{n:>11}" for n, _ in PHASES))
print(f"  {'':31}" + "".join(f"{'waste / fb':>11}" for _ in PHASES))
for label, policy in POLICIES:
    cells = "".join(f"{f'{w * 100:.0f} / {f * 100:.0f}':>11}" for _, w, f in simulate(policy))
    print(f"  {label:31}{cells}")
print("\nwaste: % of requests that failed on the primary first (each costs its latency)")
print("fb: % answered by the fallback model (each is a different model's answer)")
