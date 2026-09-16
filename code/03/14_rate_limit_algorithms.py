# How a server decides to say 429. Three common algorithms, each allowing "10 requests
# per second" on paper, fed the same burst of traffic.

LIMIT, WINDOW = 10, 1.0

# A client that sends 10 requests just before a window boundary and 10 just after it,
# then settles to a steady stream of one request every 50 milliseconds.
arrivals = [0.90 + i * 0.009 for i in range(10)] + [1.00 + i * 0.009 for i in range(10)]
arrivals += [1.20 + i * 0.05 for i in range(30)]


def fixed_window(times):
    counts, allowed = {}, []
    for t in times:
        window = int(t // WINDOW)
        if counts.get(window, 0) < LIMIT:
            counts[window] = counts.get(window, 0) + 1
            allowed.append(t)
    return allowed


def sliding_log(times):
    log, allowed = [], []
    for t in times:
        log = [s for s in log if s > t - WINDOW]
        if len(log) < LIMIT:
            log.append(t)
            allowed.append(t)
    return allowed


def token_bucket(times, capacity=LIMIT, rate=LIMIT / WINDOW):
    tokens, last, allowed = float(capacity), 0.0, []
    for t in times:
        tokens = min(capacity, tokens + (t - last) * rate)
        last = t
        if tokens >= 1:
            tokens -= 1
            allowed.append(t)
    return allowed


def busiest_second(allowed):
    return max(sum(1 for s in allowed if start <= s < start + WINDOW)
               for start in [t for t in allowed])


print(f"{len(arrivals)} requests arrive; the limit is {LIMIT} per {WINDOW:.0f}s\n")
print(f"  {'algorithm':14} {'allowed':>7} {'rejected':>8}   most allowed in any 1-second span")
for name, fn in [("fixed window", fixed_window), ("sliding log", sliding_log),
                 ("token bucket", token_bucket)]:
    allowed = fn(arrivals)
    print(f"  {name:14} {len(allowed):>7} {len(arrivals) - len(allowed):>8}   "
          f"{busiest_second(allowed)}")
