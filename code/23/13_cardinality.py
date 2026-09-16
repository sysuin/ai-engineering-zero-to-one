# Every distinct combination of label values on a metric is a separate time series, stored and
# indexed on its own. A simulated day of Clarity traffic, and what each extra label does to the count.

import numpy as np

rng = np.random.default_rng(23)
REQUESTS, TENANTS, USERS_PER_TENANT = 50_000, 400, 20
EDGES = 11                                   # a latency histogram with eleven bucket edges
PER_COMBINATION = EDGES + 1 + 2              # one series per bucket, plus +Inf, sum and count

routes = rng.choice(["ask", "ask_stream", "feedback", "eval", "health", "admin"], REQUESTS,
                    p=[.55, .25, .08, .05, .06, .01])
models = np.where(rng.random(REQUESTS) < 0.1, "smart", "fast")
statuses = rng.choice(["200", "400", "429", "500", "504"], REQUESTS, p=[.95, .01, .02, .01, .01])
# A few large tenants and a long tail of small ones, as real customer lists are.
weights = 1 / np.arange(1, TENANTS + 1) ** 1.1
tenants = rng.choice(TENANTS, REQUESTS, p=weights / weights.sum())
users = tenants * USERS_PER_TENANT + rng.integers(0, USERS_PER_TENANT, REQUESTS)
traces = np.arange(REQUESTS)

LABELS = {"route": routes, "model": models, "status": statuses, "tenant": tenants,
          "user": users, "trace_id": traces}


def combinations(names: list[str]) -> int:
    rows = np.stack([np.unique(LABELS[n], return_inverse=True)[1] for n in names], axis=1)
    return len(np.unique(rows, axis=0))


print(f"a day of {REQUESTS:,} requests; one latency histogram = {PER_COMBINATION} series "
      "per label combination")
print("distinct values seen: " + ", ".join(f"{n} {len(np.unique(v)):,}"
                                         for n, v in list(LABELS.items())[:4]) + ",")
print("  " + ", ".join(f"{n} {len(np.unique(v)):,}" for n, v in list(LABELS.items())[4:]))
print(f"\n  {'labels on the histogram':40}{'combinations':>13}{'series':>11}")
steps = [["route", "model", "status"], ["route", "model", "status", "tenant"],
         ["route", "model", "status", "tenant", "user"],
         ["route", "model", "status", "trace_id"]]
for names in steps:
    seen = combinations(names)
    print(f"  {' + '.join(names):40}{seen:>13,}{seen * PER_COMBINATION:>11,}")

top = np.bincount(tenants, minlength=TENANTS)
big = np.argsort(-top)[:20]
share = top[big].sum() / REQUESTS
capped = np.where(np.isin(tenants, big), tenants, -1)             # -1 stands for "other"
LABELS["tenant (top 20 + other)"] = capped
seen = combinations(["route", "model", "status", "tenant (top 20 + other)"])
print(f"  {'route + model + status + top-20 tenant':40}{seen:>13,}{seen * PER_COMBINATION:>11,}")
print(f"\nthe twenty largest tenants sent {share:.0%} of requests; the rest are 'other'")
print("a trace id as a label makes new series for every request, every day")
