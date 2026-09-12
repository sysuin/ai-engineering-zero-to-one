# Why jitter matters: sixty clients, one outage, and what happens when it ends.
#
# No network here. This is a simulation, because the point is a pattern in time and
# you cannot see a pattern in time by watching one client.

import json
import random
from pathlib import Path

CLIENTS = 60
ATTEMPTS = 5
BASE_DELAY = 1.0


def retry_times(jitter: bool, rng: random.Random) -> list[float]:
    """When each retry lands, in seconds after the outage began."""
    times, t = [], 0.0
    for attempt in range(ATTEMPTS):
        ceiling = BASE_DELAY * (2 ** attempt)
        t += rng.uniform(0, ceiling) if jitter else ceiling
        times.append(round(t, 3))
    return times


WINDOW = 0.1        # 100 milliseconds — a server feels a spike at this resolution


def busiest_window(all_times: list[list[float]]) -> tuple[float, int]:
    """
    The 100ms window in which the most retries land.

    The resolution matters. Measured by the second, the two strategies look similar.
    Measured the way a server actually experiences load, they do not.
    """
    buckets: dict[int, int] = {}
    for times in all_times:
        for t in times:
            slot = int(t / WINDOW)
            buckets[slot] = buckets.get(slot, 0) + 1
    slot = max(buckets, key=lambda k: buckets[k])
    return slot * WINDOW, buckets[slot]


recorded = {}

for jitter in (False, True):
    rng = random.Random(11)
    all_times = [retry_times(jitter, rng) for _ in range(CLIENTS)]
    recorded["jitter" if jitter else "no_jitter"] = all_times
    at, count = busiest_window(all_times)
    label = "with jitter   " if jitter else "without jitter"

    print(f"{label}  worst 100ms window: t={at:.1f}s takes {count} requests at once")
    print(f"                  first client retries at "
          f"{', '.join(f'{t:.2f}s' for t in all_times[0])}")

# Keep the timings so the book can plot exactly what this run produced.
Path("code/03/_jitter_times.json").write_text(json.dumps(recorded))

print()
print("Without jitter every client waits exactly the same 1s, 2s, 4s ... so all sixty")
print("come back at the same five instants, and the server that has just recovered is")
print("knocked over again. Jitter spreads the identical amount of load across the")
print("identical window, and the spike disappears.")
