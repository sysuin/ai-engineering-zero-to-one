# Three rate limiters with the same nominal limit — ten requests a second — and the most
# requests each one admits in any single second, under an adversarial burst pattern.

import collections

LIMIT, WINDOW = 10, 1.0


class FixedWindow:
    """Count requests per calendar second. Simple, and generous at the boundaries."""
    def __init__(self):
        self.window, self.count = None, 0

    def allow(self, t: float) -> bool:
        window = int(t // WINDOW)
        if window != self.window:
            self.window, self.count = window, 0
        if self.count < LIMIT:
            self.count += 1
            return True
        return False


class SlidingLog:
    """Remember every admitted request in the last second. Exact, and memory per request."""
    def __init__(self):
        self.admitted = collections.deque()

    def allow(self, t: float) -> bool:
        while self.admitted and self.admitted[0] <= t - WINDOW:
            self.admitted.popleft()
        if len(self.admitted) < LIMIT:
            self.admitted.append(t)
            return True
        return False


class TokenBucket:
    """Refill at LIMIT a second, hold at most `burst`. Two numbers of state per client."""
    def __init__(self, burst: int = LIMIT):
        self.burst, self.tokens, self.last = burst, float(burst), 0.0

    def allow(self, t: float) -> bool:
        self.tokens = min(self.burst, self.tokens + (t - self.last) * LIMIT)
        self.last = t
        if self.tokens >= 1:
            self.tokens -= 1
            return True
        return False


# The attack on a fixed window: a burst just before a boundary and another just after.
arrivals = sorted([0.95 + i * 0.004 for i in range(12)] + [1.0 + i * 0.004 for i in range(12)]
                  + [3.0 + i * 0.1 for i in range(40)])


def worst_second(times: list[float]) -> int:
    return max(sum(1 for u in times if t <= u < t + WINDOW) for t in times)


print(f"nominal limit: {LIMIT} requests per second; {len(arrivals)} requests arrive\n")
print(f"  {'limiter':<24} {'admitted':>9} {'most in one second':>19}   state kept")
for name, limiter, state in (("fixed window", FixedWindow(), "one counter"),
                             ("sliding log", SlidingLog(), "a timestamp per request"),
                             ("token bucket, burst 10", TokenBucket(10), "two numbers"),
                             ("token bucket, burst 3", TokenBucket(3), "two numbers")):
    admitted = [t for t in arrivals if limiter.allow(t)]
    print(f"  {name:<24} {len(admitted):>9} {worst_second(admitted):>19}   {state}")

print("\nThe fixed window admitted twice its limit inside one second, by taking a full")
print("allowance on each side of a boundary. The sliding log never exceeds the limit and")
print("pays with memory; the token bucket is close to it with two numbers of state, and its")
print("burst size is the dial between absorbing a spike and smoothing it.")
