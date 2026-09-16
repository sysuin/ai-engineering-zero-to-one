# Depends on code/21/_scorecard.json from 01; re-run after it changes.
# The golden set's mix of kinds is not your traffic's mix. Re-weighting the per-kind scores
# by the traffic estimates what users will see, and says which slice the estimate is least
# sure about.

import collections
import json
import math

rows = json.load(open("code/21/_scorecard.json"))
by_kind = collections.defaultdict(list)
for row in rows:
    by_kind[row["kind"]].append(row["correct"])

# An assumed traffic mix, for illustration: in a real system it comes from classifying a
# sample of production questions into the same kinds.
TRAFFIC = {"document": 0.45, "warehouse": 0.35, "unanswerable": 0.12, "composite": 0.08}

n_total = len(rows)
print(f"{n_total} golden cases, scored once, against an assumed traffic mix\n")
print(f"  {'kind':<14} {'cases':>6} {'score':>7} {'share of set':>13} {'share of traffic':>17}")
for kind in TRAFFIC:
    scores = by_kind[kind]
    print(f"  {kind:<14} {len(scores):>6} {sum(scores) / len(scores):>7.0%} "
          f"{len(scores) / n_total:>13.0%} {TRAFFIC[kind]:>17.0%}")


def weighted(weights: dict[str, float]) -> tuple[float, float, dict[str, float]]:
    """Post-stratified estimate, its standard error, and each slice's share of the variance."""
    estimate = sum(w * sum(by_kind[k]) / len(by_kind[k]) for k, w in weights.items())
    parts = {}
    for k, w in weights.items():
        p, n = sum(by_kind[k]) / len(by_kind[k]), len(by_kind[k])
        # A slice with every case right still has uncertainty; the (x + 1) / (n + 2) form
        # keeps it from reporting none.
        p_smoothed = (sum(by_kind[k]) + 1) / (n + 2)
        parts[k] = w * w * p_smoothed * (1 - p_smoothed) / n
    return estimate, math.sqrt(sum(parts.values())), parts


set_mix = {k: len(by_kind[k]) / n_total for k in TRAFFIC}
raw, raw_se, _ = weighted(set_mix)
est, se, parts = weighted(TRAFFIC)
print(f"\n  score as the set is mixed          {raw:.0%}  (± {1.96 * raw_se:.0%})")
print(f"  score re-weighted to the traffic   {est:.0%}  (± {1.96 * se:.0%})")
biggest = max(parts, key=parts.get)
print(f"\n  {parts[biggest] / sum(parts.values()):.0%} of the uncertainty in the re-weighted "
      f"figure comes from '{biggest}',")
print(f"  which has {len(by_kind[biggest])} cases and {TRAFFIC[biggest]:.0%} of the traffic.")

# Where should cases go? Neyman allocation gives each slice cases in proportion to its
# traffic share times its standard deviation.
weights = {k: TRAFFIC[k] * math.sqrt(((sum(by_kind[k]) + 1) / (len(by_kind[k]) + 2))
                                     * (1 - (sum(by_kind[k]) + 1) / (len(by_kind[k]) + 2)))
           for k in TRAFFIC}
total = sum(weights.values())
print("\n  a set of 100 cases, allocated to make the re-weighted figure as precise as possible:")
for k in TRAFFIC:
    print(f"    {k:<14} {round(100 * weights[k] / total):>3}")
