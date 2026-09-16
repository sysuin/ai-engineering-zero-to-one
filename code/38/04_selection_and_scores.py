# Three things an eval report gets wrong after the interval is right: picking the best of
# many variants on the set that scored them, reading a p-value as the chance of being right,
# and averaging scores that are not pass or fail. No model calls; simulation and arithmetic.

import math
import random
import statistics

rng = random.Random(38)


# ------------------------------------------------------------------ 1. the best of twenty
def best_of(variants: int, cases: int, true_rate: float, shared: bool) -> tuple[float, float]:
    """Score every variant on one development set, keep the best, then score it afresh."""
    if shared:
        # Cases differ in difficulty and every variant meets the same cases, so variants pass
        # and fail together — which is how prompt variants on one set behave.
        ease = [rng.betavariate(true_rate * 4, (1 - true_rate) * 4) for _ in range(cases)]
    else:
        ease = [true_rate] * cases
    dev = [sum(rng.random() < e for e in ease) / cases for _ in range(variants)]
    fresh_ease = ([rng.betavariate(true_rate * 4, (1 - true_rate) * 4) for _ in range(cases)]
                  if shared else [true_rate] * cases)
    fresh = sum(rng.random() < e for e in fresh_ease) / cases
    return max(dev), fresh


print("Twenty prompt variants that are all truly 80% accurate, each scored on 120 cases.")
print("Keep the best, then score it on 120 new cases. Mean of 2,000 repetitions:\n")
print(f"  {'cases':<34}{'best on dev':>12}{'same prompt, new cases':>24}")
for label, shared in (("independent", False), ("shared difficulty", True)):
    runs = [best_of(20, 120, 0.80, shared) for _ in range(2_000)]
    print(f"  {label:<34}{statistics.mean(r[0] for r in runs):>12.1%}"
          f"{statistics.mean(r[1] for r in runs):>24.1%}")
for variants in (1, 5, 50):
    runs = [best_of(variants, 120, 0.80, False) for _ in range(2_000)]
    print(f"  {f'independent, best of {variants}':<34}{statistics.mean(r[0] for r in runs):>12.1%}"
          f"{statistics.mean(r[1] for r in runs):>24.1%}")


# ------------------------------------------------------------------ 2. a Bayesian reading
def mcnemar_p(b: int, c: int) -> float:
    n, k = b + c, min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


ONLY_A, ONLY_B = 11, 5                   # the paired comparison in 01_intervals.py
draws = [rng.betavariate(ONLY_A + 1, ONLY_B + 1) for _ in range(200_000)]
p_a_better = sum(d > 0.5 for d in draws) / len(draws)
print(f"\nThe paired comparison from the first listing: A alone right on {ONLY_A}, "
      f"B alone on {ONLY_B}\n")
print(f"  McNemar's exact test, two-sided          p = {mcnemar_p(ONLY_A, ONLY_B):.3f}")
print(f"  one-sided (is A better?)                 p = {mcnemar_p(ONLY_A, ONLY_B) / 2:.3f}")
print(f"  flat prior: P(A wins more disagreements) = {p_a_better:.3f}")
for strength in (10, 40):
    sceptic = [rng.betavariate(ONLY_A + strength, ONLY_B + strength) for _ in range(200_000)]
    print(f"  sceptical prior worth {strength * 2} disagreements  "
          f"= {sum(d > 0.5 for d in sceptic) / len(sceptic):.3f}")


# ------------------------------------------------------------------ 3. scores from 1 to 5
# A judge scores sixty answers from each version. B nudges most answers up a point and
# wrecks a few; the question you ask of the scores decides what you conclude.
cases = 60
a_scores = [rng.choice([3, 4, 4, 5]) for _ in range(cases)]
b_scores = []
for s in a_scores:
    roll = rng.random()
    if roll < 0.10:
        b_scores.append(max(1, s - 3))           # a few answers ruined
    elif roll < 0.60:
        b_scores.append(min(5, s + 1))           # half improved by a point
    else:
        b_scores.append(s)
diffs = [b - a for a, b in zip(a_scores, b_scores)]
mean_diff = statistics.mean(diffs)
se = statistics.stdev(diffs) / math.sqrt(cases)
z = mean_diff / se
p_t = 2 * (1 - statistics.NormalDist().cdf(abs(z)))
better, worse = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
boot = sorted(statistics.mean(rng.choices(diffs, k=cases)) for _ in range(10_000))

print(f"\nA judge's 1-5 scores for the same {cases} answers from A and B\n")
print(f"  mean score                 A {statistics.mean(a_scores):.2f}   B {statistics.mean(b_scores):.2f}")
print(f"  mean difference (B - A)    {mean_diff:+.2f}, 95% bootstrap interval "
      f"{boot[250]:+.2f} to {boot[9750]:+.2f}")
print(f"  paired test on the means   p = {p_t:.3f} (normal approximation)")
print(f"  answers B made better      {better}, worse {worse}, "
      f"sign test p = {mcnemar_p(better, worse):.4f}")
print(f"  answers B scored 1 or 2    {sum(b <= 2 for b in b_scores)}  (A: {sum(a <= 2 for a in a_scores)})")
