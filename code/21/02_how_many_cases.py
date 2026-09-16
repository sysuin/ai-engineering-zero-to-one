# timeout: 300
# Reads code/21/_scorecard.json, written by 01 — this file resamples those
# outcomes and never calls a model.
# "Is 20 enough?" is a question with an arithmetic answer.

import json
import random
from statistics import mean

random.seed(21)
BOOTSTRAPS = 2000
rows = json.load(open("code/21/_scorecard.json"))
outcomes = [1 if r["correct"] else 0 for r in rows]
true_score = mean(outcomes)


def interval(n: int) -> tuple[float, float, float]:
    """
    Resample n cases from the run we already did, two thousand times.

    This is a bootstrap: it asks what score you would have reported if your eval set
    had been n cases drawn from the same pool. The spread of those answers is the
    honest error bar on a set of that size.
    """
    scores = sorted(mean(random.choices(outcomes, k=n)) for _ in range(BOOTSTRAPS))
    return (scores[int(0.025 * BOOTSTRAPS)], scores[int(0.975 * BOOTSTRAPS)],
            mean(scores))


print(f"The measured score on all {len(outcomes)} cases is {true_score:.0%}.")
print(f"Here is what smaller eval sets would have told you, "
      f"{BOOTSTRAPS:,} resamples each.\n")
print(f"  {'cases':>6}  {'95% of runs report':>22}   width")
sizes = [5, 10, 20, 40, 80, 120, 300, 1000]
results = {}
for n in sizes:
    low, high, _ = interval(n)
    results[n] = {"low": low, "high": high, "width": high - low}
    bar = "-" * max(1, round((high - low) * 60))
    print(f"  {n:>6}  {low:>10.0%} to {high:<10.0%} {high - low:>5.0%}  {bar}")

json.dump({"true": true_score, "n": len(outcomes), "results": results},
          open("code/21/_power.json", "w"), indent=1)

twenty = results[20]["width"]
hundred = results[120]["width"]
print()
print(f"A 20-case eval set reports a number that is {twenty:.0%} wide. That is not a")
print("measurement, it is a mood. Two systems five points apart are indistinguishable")
print("to it, and five points is the size of most real improvements.")
print()
print(f"At {len(outcomes)} cases the interval is {hundred:.0%} wide, which is enough to "
      f"tell a")
print("ten-point change from noise and not enough to tell a three-point one. That is")
print("the honest position of this chapter's eval set, and it is why §21.14 talks")
print("about growing it rather than finishing it.")
print()
print("Note what the last two rows cost: nothing to compute here, and a thousand model")
print("runs to actually have. Before paying for them, be clear about what size of")
print("difference you need to detect — that number, not a round one, decides how many")
print("cases you need.")
