# Routing, measured on the benchmark's own cases rather than argued about: the cheap model
# always, the smart model always, and a cascade that escalates only when the cheap model
# declines to answer. No new calls — every number comes from 03's per-case records.

import json
import sys

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST, MODEL_SMART                # noqa: E402

# Placeholder rates, dollars per million tokens in and out: the ratio between the tiers is
# what matters, and Appendix C says where to find real ones.
RATES = {MODEL_FAST: (0.15, 0.60), MODEL_SMART: (2.00, 8.00)}

bench = json.load(open("code/27/_bench.json"))
cheap = next(r for r in bench if r["model"] == MODEL_FAST)
strong = next(r for r in bench if r["model"] == MODEL_SMART and r["ceiling"] > 250)
by_id = {c["id"]: c for c in strong["cases"]}


def cost(case: dict, model: str) -> float:
    rate_in, rate_out = RATES[model]
    return (case["tokens_in"] * rate_in + case["tokens_out"] * rate_out) / 1e6


policies = {"cheap model always": [], "smart model always": [],
            "cascade: escalate on refusal": [], "best of both (no router can)": []}
for first in cheap["cases"]:
    second = by_id[first["id"]]
    escalate = first["abstained"]
    policies["cheap model always"].append((first["correct"], cost(first, MODEL_FAST),
                                           first["kind"]))
    policies["smart model always"].append((second["correct"], cost(second, MODEL_SMART),
                                           first["kind"]))
    policies["cascade: escalate on refusal"].append(
        (second["correct"] if escalate else first["correct"],
         cost(first, MODEL_FAST) + (cost(second, MODEL_SMART) if escalate else 0),
         first["kind"]))
    policies["best of both (no router can)"].append(
        (first["correct"] or second["correct"], float("nan"), first["kind"]))

escalated = sum(c["abstained"] for c in cheap["cases"])
baseline = sum(c for _, c, _ in policies["smart model always"])
print(f"{len(cheap['cases'])} cases; the cascade escalated {escalated} of them\n")
print(f"  {'policy':<30} {'correct':>8} {'answerable':>11} {'unanswerable':>13} "
      f"{'cost':>6}")
for name, rows in policies.items():
    answerable = [r for r in rows if r[2] != "unanswerable"]
    unanswerable = [r for r in rows if r[2] == "unanswerable"]
    total = sum(c for _, c, _ in rows)
    shown = "  —" if total != total else f"{total / baseline:>5.0%}"
    print(f"  {name:<30} {sum(r[0] for r in rows):>5}/{len(rows):<2} "
          f"{sum(r[0] for r in answerable):>8}/{len(answerable):<2} "
          f"{sum(r[0] for r in unanswerable):>10}/{len(unanswerable):<2} {shown:>6}")

print("\nCost is relative to sending everything to the smart model, at placeholder rates.")
print("A refusal is the only signal here that needs no second model to read, and on the")
print("unanswerable cases it escalates exactly the questions the cheap model got right —")
print("so read that column before believing the cascade's total.")
