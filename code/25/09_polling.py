# A job API returns 202 and a URL, and the client asks until the work is done. How often should
# it ask? Three polling policies against the same thousand jobs, measured in requests and delay.

import random
import statistics

rng = random.Random(25)
JOBS = [rng.lognormvariate(2.3, 0.8) for _ in range(1000)]     # seconds of work: median about 10s


def fixed(interval):
    def policy(elapsed, _estimate):
        return interval
    return policy


def backoff(first=1.0, factor=1.5, ceiling=30.0):
    def policy(elapsed, _estimate):
        return min(ceiling, first * factor ** policy.polls)
    policy.polls = 0
    return policy


def retry_after(elapsed, estimate):
    """The server says how long the rest should take, from what it knows about this kind of job."""
    return max(1.0, estimate - elapsed)


def run(make_policy, estimate_error=0.3):
    polls, delays = [], []
    for duration in JOBS:
        policy = make_policy()
        if hasattr(policy, "polls"):
            policy.polls = 0
        estimate = duration * rng.uniform(1 - estimate_error, 1 + estimate_error)
        t, n = 0.0, 0
        while True:
            wait = policy(t, estimate)
            if hasattr(policy, "polls"):
                policy.polls += 1
            t += wait
            n += 1
            if t >= duration:
                break
        polls.append(n)
        delays.append(t - duration)
    return statistics.mean(polls), statistics.median(delays), sorted(delays)[int(0.95 * len(delays))]


print(f"1,000 jobs, median {statistics.median(JOBS):.1f}s of work, the slowest tenth over "
      f"{sorted(JOBS)[900]:.0f}s\n")
print(f"  {'policy':34}{'polls per job':>14}{'extra wait, p50':>17}{'p95':>7}")
for label, make in (("every second", lambda: fixed(1.0)),
                    ("every five seconds", lambda: fixed(5.0)),
                    ("backoff from 1s, x1.5, max 30s", backoff),
                    ("Retry-After, estimate ±30%", lambda: retry_after)):
    polls, p50, p95 = run(make)
    print(f"  {label:34}{polls:>14.1f}{p50:>16.1f}s{p95:>6.1f}s")
