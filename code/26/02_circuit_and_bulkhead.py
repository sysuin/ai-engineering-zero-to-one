# timeout: 900
# What a retry loop does to an outage, and what stops it.

import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.platform.resilience import (Bulkhead, CircuitBreaker,   # noqa: E402
                                         Open, Retry, Transient)

random.seed(26)
CALLS = 60
LATENCY = 0.05          # what a healthy call costs
OUTAGE_LATENCY = 0.30   # what a failing provider costs before it errors


class Provider:
    """A dependency that is down for the middle third of the window."""

    def __init__(self) -> None:
        self.attempts = 0
        self.down = False

    def __call__(self) -> str:
        self.attempts += 1
        if self.down:
            time.sleep(OUTAGE_LATENCY)      # failures are slower than successes
            raise Transient("provider unavailable")
        time.sleep(LATENCY)
        return "ok"


def window(strategy) -> dict:
    provider = Provider()
    started = time.perf_counter()
    ok = failed = shed = 0
    for n in range(CALLS):
        provider.down = CALLS // 3 <= n < 2 * CALLS // 3
        try:
            strategy(provider)
            ok += 1
        except Open:
            shed += 1
        except Exception:
            failed += 1
    return {"seconds": time.perf_counter() - started, "ok": ok, "failed": failed,
            "shed": shed, "upstream_calls": provider.attempts}


retry = Retry(attempts=4)
breaker = CircuitBreaker(threshold=3, cooldown=0.5)

strategies = {
    "no protection": lambda p: p(),
    "retry only": lambda p: retry(p),
    "retry + circuit breaker": lambda p: breaker(lambda: retry(p)),
}

print(f"{CALLS} calls; the provider is down for the middle third.\n")
print(f"  {'':<26}{'seconds':>9}{'ok':>6}{'failed':>8}{'shed':>6}"
      f"{'upstream calls':>16}")
results = {}
for label, strategy in strategies.items():
    row = window(strategy)
    results[label] = row
    print(f"  {label:<26}{row['seconds']:>8.1f}s{row['ok']:>6}{row['failed']:>8}"
          f"{row['shed']:>6}{row['upstream_calls']:>16}")

json.dump(results, open("code/26/_resilience.json", "w"), indent=1)

bare, retried, guarded = (results["no protection"], results["retry only"],
                          results["retry + circuit breaker"])
print()
print(f"Read the last column first. A retry loop turned {bare['upstream_calls']} "
      f"calls into "
      f"{retried['upstream_calls']},")
print("and every one of those extra calls landed on a provider that was already")
print("failing. That is the retry stampede: the moment a dependency wobbles, its")
print("load goes up rather than down.")
print()
print(f"The breaker cut it to {guarded['upstream_calls']} and the window from "
      f"{retried['seconds']:.0f}s to {guarded['seconds']:.0f}s, by giving up")
print(f"quickly instead of slowly — {guarded['shed']} calls were shed without being "
      f"tried at all.")
print()
print(f"Now read the cost, because there is one. Successes fell from "
      f"{bare['ok']} to {guarded['ok']}: the")
print(f"breaker sheds every call while it is open, and some of those would have")
print("succeeded — the provider came back before the cooldown expired and the")
print("breaker did not know yet.")
print()
print("That is the trade, and it is not optional. A breaker converts slow failures")
print("into fast ones and pays for it with false negatives during recovery. Tune the")
print("cooldown too long and you are down after the provider is up; too short and")
print("you are back to the stampede.")

# ------------------------------------------------------------------ bulkhead
print("\n--- and the failure mode a breaker does not catch ---\n")


def slow() -> str:
    time.sleep(0.4)        # not broken. Just slow. Nothing to trip a breaker on.
    return "ok"


def quick() -> str:
    time.sleep(0.01)       # a different endpoint, with nothing to do with the first
    return "ok"


def mixed(guard) -> dict:
    """
    Twelve calls to a slow dependency and eight to a fast one, sharing six workers.

    The question is not whether the slow calls are slow. It is what happens to the
    eight that had nothing to do with them.
    """
    started = time.perf_counter()
    outcomes = {"slow_ok": 0, "shed": 0, "quick_latencies": []}

    def call(kind):
        if kind == "slow":
            try:
                guard(slow)
                outcomes["slow_ok"] += 1
            except Open:
                outcomes["shed"] += 1
        else:
            # Latency measured from submission, not from when a worker picks it up —
            # the waiting is the damage, and a timer started inside the handler is
            # blind to it.
            quick()
            outcomes["quick_latencies"].append(time.perf_counter() - started)

    # The burst arrives slow-first, which is what a burst looks like.
    work = ["slow"] * 12 + ["quick"] * 8
    with ThreadPoolExecutor(max_workers=6) as pool:
        futures = [pool.submit(call, kind) for kind in work]
        started_at = {}
        for future in futures:
            future.result()
    latencies = sorted(outcomes.pop("quick_latencies"))
    return {"seconds": time.perf_counter() - started,
            "quick_p95": latencies[int(0.95 * (len(latencies) - 1))] if latencies
            else 0.0,
            "quick_waited": time.perf_counter() - started, **outcomes}


plain = mixed(lambda fn: fn())
walled = mixed(Bulkhead(limit=2))
print("  twenty calls over six workers: twelve to a slow dependency, eight to a")
print("  fast endpoint that has nothing to do with it\n")
print(f"  {'':<26}{'window':>9}{'slow ok':>9}{'shed':>6}"
      f"{'fast p95':>11}")
for label, row in (("every worker available", plain), ("bulkhead of 2", walled)):
    print(f"  {label:<26}{row['seconds']:>8.1f}s{row['slow_ok']:>9}{row['shed']:>6}"
          f"{row['quick_p95'] * 1000:>10.0f}ms")
json.dump({"plain": plain, "walled": walled},
          open("code/26/_bulkhead.json", "w"), indent=1)

print()
print("Nothing errored in either run, so no breaker would have tripped. The")
print("dependency is not broken — it has become slow, which is more common and more")
print("dangerous, because a slow dependency quietly consumes every worker in the")
print("process and takes down the endpoints that do not use it at all.")
print()
print(f"Read the last column. The fast endpoint's p95 went from "
      f"{plain['quick_p95'] * 1000:.0f}ms to "
      f"{walled['quick_p95'] * 1000:.0f}ms —")
print(f"{plain['quick_p95'] / max(walled['quick_p95'], 1e-9):.0f}x — and it has "
      f"nothing to do with the slow dependency. Without the bulkhead it")
print("spent its life queued behind twelve calls it never made.")
print()
print(f"The bulkhead paid for that by shedding {walled['shed']} calls to the slow "
      f"dependency. That is a")
print("real cost and the entire point: you have decided, in advance, which work is")
print("allowed to consume the process. Decide it in advance or the slowest dependency")
print("decides it for you.")
