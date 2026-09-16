# A provider can fail after a stream has started, when the person has already read part of the
# answer. Four policies for a gateway, simulated under two assumptions about when streams fail.
# Every number below is an assumption stated here, not a measurement of any provider.

import random

TTFT, RATE = 0.8, 40.0                # primary: seconds to first token, tokens a second
TTFT_FALLBACK, RATE_FALLBACK = 1.2, 30.0
LENGTH = 400                          # tokens in an answer: ten seconds of streaming
FAILS = 0.03                          # share of streams that fail after the request is accepted
REQUESTS = 100_000
DURATION = TTFT + LENGTH / RATE


def failure_time(shape: str, rng: random.Random) -> float | None:
    if rng.random() >= FAILS:
        return None
    if shape == "anywhere in the stream":
        return rng.uniform(0, DURATION)
    return min(rng.expovariate(1 / 0.5), DURATION)       # "mostly at the start": mean 0.5s


def one_request(policy: str, hold: float, fails_at: float | None):
    """First text and whole answer, in seconds, and what was seen."""
    shown_at = TTFT + hold      # nothing shows until the hold ends
    if fails_at is None:
        return shown_at, DURATION, "fine"
    if policy == "pass the error on":
        seen = shown_at if fails_at > shown_at else None
        return seen, None, "broken"
    done = fails_at + TTFT_FALLBACK + LENGTH / RATE_FALLBACK
    if fails_at < shown_at:     # nothing was shown yet
        return fails_at + TTFT_FALLBACK, done, "invisible"
    return shown_at, done, "retracted"      # withdrawn, restarted


def percentile(values: list[float], share: float) -> float:
    values = sorted(values)
    return values[int(share * (len(values) - 1))]


POLICIES = [("pass the error on", 0.0), ("restart, retracting", 0.0),
            ("hold back 1 second", 1.0), ("hold back 3 seconds", 3.0),
            ("hold back the whole answer", LENGTH / RATE)]
print(f"an answer streams for {DURATION:.0f}s; {FAILS:.0%} of streams fail part-way; "
      f"{REQUESTS:,} requests\n")
for shape in ("anywhere in the stream", "mostly at the start"):
    print(f"failures {shape}")
    print(f"  {'policy':<28}{'first text':>11}{'p99 done':>10}{'broken':>8}{'retracted':>11}"
          f"{'invisible':>10}")
    for name, hold in POLICIES:
        rng = random.Random(27)
        policy = "pass the error on" if name == "pass the error on" else "fail over"
        results = [one_request(policy, hold, failure_time(shape, rng)) for _ in range(REQUESTS)]
        first = [r[0] for r in results if r[0] is not None]
        done = [r[1] for r in results if r[1] is not None]
        seen = {k: sum(r[2] == k for r in results) / REQUESTS for k in ("broken", "retracted",
                                                                       "invisible")}
        print(f"  {name:<28}{percentile(first, 0.5):>10.1f}s{percentile(done, 0.99):>9.1f}s"
              f"{seen['broken']:>8.1%}{seen['retracted']:>11.1%}{seen['invisible']:>10.1%}")
    print()
print("first text = median seconds until the person sees any of the answer")
