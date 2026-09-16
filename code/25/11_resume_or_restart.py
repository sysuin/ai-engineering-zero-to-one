# A stream that drops is not a client that left, and the server cannot tell them apart when the
# connection closes. Three policies for a streamed agent run, simulated. Every rate below is an
# assumption stated here; change them to your own traffic.

import random

STEPS, STEP_SECONDS = 8, 3.0         # a run of eight model calls, three seconds each
RECONNECT_AFTER = 2.0                # a dropped client is back this many seconds later
GRACE = 6.0                          # how long the resuming server keeps a disconnected run
ABANDON = 0.15                       # share of people who close the tab at some point
QUESTIONS = 20_000


def one_question(policy: str, drops_per_minute: float, rng: random.Random):
    """Model calls made, calls made for nobody, and seconds until the answer (None if abandoned)."""
    leaves_at = rng.uniform(0, STEPS * STEP_SECONDS) if rng.random() < ABANDON else None
    calls = wasted = 0
    done_steps, clock = 0, 0.0
    while done_steps < STEPS:
        step_end = clock + STEP_SECONDS
        drop = clock + rng.expovariate(drops_per_minute / 60) if drops_per_minute else float("inf")
        gone = leaves_at is not None and leaves_at < step_end
        if gone:                                   # the person has left for good
            if policy == "never stop":
                calls += STEPS - done_steps
                wasted += STEPS - done_steps
            elif policy == "keep for a grace period":
                spent = min(STEPS - done_steps, 1 + int(GRACE // STEP_SECONDS))
                calls += spent
                wasted += spent
            else:                                  # stop on disconnect: the step in progress finishes
                calls += 1
                wasted += 1
            return calls, wasted, None
        if drop < step_end:                        # the connection drops mid-step, the person stays
            lost = policy == "stop on disconnect" or (
                policy == "keep for a grace period" and RECONNECT_AFTER > GRACE)
            if lost:
                calls += 1                         # the step in progress finishes for nobody
                wasted += done_steps + 1           # and everything so far must be redone
                done_steps, clock = 0, step_end + RECONNECT_AFTER
                continue
            # never stop, or keep for a grace period: the run carries on and the client resumes
        calls += 1
        done_steps += 1
        clock = step_end
    return calls, wasted, clock


print(f"a run of {STEPS} model calls over {STEPS * STEP_SECONDS:.0f}s; {ABANDON:.0%} of people leave part-way;")
print(f"a dropped client returns after {RECONNECT_AFTER:.0f}s; the grace period is {GRACE:.0f}s\n")
print(f"  {'drops a minute':<16}{'policy':<26}{'calls':>6}{'wasted':>11}{'answer p50':>11}{'p90':>6}")
for rate in (0.1, 1.0, 4.0):
    for policy in ("stop on disconnect", "keep for a grace period", "never stop"):
        rng = random.Random(25)
        results = [one_question(policy, rate, rng) for _ in range(QUESTIONS)]
        calls = sum(r[0] for r in results) / QUESTIONS
        wasted = sum(r[1] for r in results) / sum(r[0] for r in results)
        times = sorted(r[2] for r in results if r[2] is not None)
        p50, p90 = times[len(times) // 2], times[int(len(times) * 0.9)]
        print(f"  {rate:<16}{policy:<26}{calls:>6.2f}{wasted:>11.0%}{p50:>10.0f}s{p90:>5.0f}s")
    print()
print("  wasted = calls made after the person left, or redone after a dropped connection")
