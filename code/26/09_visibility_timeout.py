# "Visibility timeouts must exceed the work", simulated. A queue hides a message from other
# workers for a while after one takes it; if the work is not acknowledged by then, the message
# reappears and another worker starts it again. Two hundred five-minute jobs and eight workers,
# arriving all at once (a backlog) or one every two minutes (a trickle).

import heapq

import numpy as np

JOBS, WORKERS, MEAN = 200, 8, 300.0            # seconds of work per job


def run(visibility: float, heartbeat: bool, gap: float, seed: int = 26):
    """Returns (hours to drain, executions per job, most copies of one job at once)."""
    rng = np.random.default_rng(seed)
    durations = rng.uniform(0.8 * MEAN, 1.2 * MEAN, JOBS)
    visible_at = [j * gap for j in range(JOBS)]  # when each unfinished message can be taken
    done = [False] * JOBS
    running: list[tuple[float, int]] = []      # (finish time, job) per busy worker
    executions, now = 0, 0.0
    copies = [0] * JOBS
    most = 0
    while not all(done):
        # free workers take the earliest visible, unfinished message
        while len(running) < WORKERS:
            ready = [j for j in range(JOBS)
                     if not done[j] and visible_at[j] <= now]
            if not ready:
                break
            j = min(ready, key=lambda k: visible_at[k])
            executions += 1
            copies[j] += 1
            most = max(most, copies[j])
            # a heartbeat keeps extending the timeout while the worker works
            hidden = durations[j] + visibility if heartbeat else visibility
            visible_at[j] = now + hidden
            heapq.heappush(running, (now + durations[j], j))
        next_visible = min((visible_at[j] for j in range(JOBS)
                            if not done[j] and visible_at[j] > now), default=np.inf)
        next_finish = running[0][0] if running else np.inf
        now = min(next_visible, next_finish)
        while running and running[0][0] <= now:
            _, j = heapq.heappop(running)
            copies[j] -= 1
            done[j] = True                     # the first to finish acknowledges; later copies are waste
    return now / 3600, executions / JOBS, most


print(f"{JOBS} jobs of about {MEAN / 60:.0f} minutes each, {WORKERS} workers\n")
print(f"  {'arrival':10}{'visibility timeout':28}{'hours':>7}{'runs per job':>14}{'most at once':>14}")
for arrival, gap in (("backlog", 0.0), ("trickle", 120.0)):
    for label, visibility, heartbeat in (("30 seconds", 30, False), ("3 minutes", 180, False),
                                         ("6 minutes", 360, False),
                                         ("30 seconds, with heartbeat", 30, True)):
        hours, per_job, most = run(visibility, heartbeat, gap)
        print(f"  {arrival:10}{label:28}{hours:>7.1f}{per_job:>14.2f}{most:>14}")
        arrival = ""
print("\nhours: until the last job is acknowledged")
print("runs per job: executions, duplicates included")
