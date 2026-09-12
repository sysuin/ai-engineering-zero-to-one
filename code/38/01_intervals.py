# How wide is your eval score, really?
#
# No model calls. Everything here is arithmetic over a score you already have, and it is
# the arithmetic that decides whether an argument about three points is worth having.

import math
import random

random.seed(7)


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """
    A 95% interval for a proportion that behaves sensibly near 0 and 1.

    The textbook interval — p ± 1.96·sqrt(p(1-p)/n) — goes above 1.0 at high scores and
    below 0.0 at low ones, which is exactly where eval scores live.
    """
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def bootstrap(results: list[int], rounds: int = 10_000) -> tuple[float, float]:
    """
    Resample your own cases, with replacement, and look at the spread.

    Makes no assumption about the shape of the distribution, which matters once your
    score is not a simple proportion — a mean of per-case scores, say, or a weighted
    blend across populations.
    """
    n = len(results)
    means = sorted(sum(random.choices(results, k=n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


SCORE = 0.85
print(f"A score of {SCORE:.0%}, at five different sample sizes.\n")
print(f"  {'n':>5}  {'passed':>6}  {'95% interval':<16}{'width':>7}   "
      "can it see a 3-point gain?")
for n in (20, 50, 120, 500, 2000):
    passed = round(SCORE * n)
    lo, hi = wilson(passed, n)
    width = (hi - lo) * 100
    verdict = "yes" if width < 6 else "no"
    print(f"  {n:>5}  {passed:>6}  {lo:.1%} - {hi:.1%}     {width:>5.1f}pt   {verdict}")

print("\nThe interval shrinks with the square root of n, which is the single most")
print("expensive fact in evaluation: to halve the width you need four times the cases.")

# ------------------------------------------------------------------ the two of them agree
n = 120
passed = round(SCORE * n)
results = [1] * passed + [0] * (n - passed)
wlo, whi = wilson(passed, n)
blo, bhi = bootstrap(results)
print(f"\nAt n={n}, the two methods agree to within a point, as they should:\n")
print(f"  Wilson      {wlo:.1%} - {whi:.1%}")
print(f"  bootstrap   {blo:.1%} - {bhi:.1%}")
print("\nUse Wilson for a simple pass rate. Use the bootstrap when the score is anything")
print("more complicated than counting successes, because it does not care what shape the")
print("statistic is.")

# ------------------------------------------------------------------ is A better than B?
def two_proportion_p(a_hits, a_n, b_hits, b_n) -> float:
    """A two-sided z test for two independent proportions."""
    pa, pb = a_hits / a_n, b_hits / b_n
    pooled = (a_hits + b_hits) / (a_n + b_n)
    se = math.sqrt(pooled * (1 - pooled) * (1 / a_n + 1 / b_n))
    if se == 0:
        return 1.0
    z = abs(pa - pb) / se
    return math.erfc(z / math.sqrt(2))          # 2 * (1 - Phi(z))


print("\n\nIs 84% better than 81%? It depends entirely on n.\n")
print(f"  {'n each':>7}  {'A':<16}{'B':<16}{'overlap?':<10}{'proper test':<14}verdict")
for n in (50, 120, 400, 1500, 3000, 6000):
    ah, bh = round(0.84 * n), round(0.81 * n)
    alo, ahi = wilson(ah, n)
    blo2, bhi2 = wilson(bh, n)
    overlap = alo < bhi2
    p_value = two_proportion_p(ah, n, bh, n)
    verdict = "A is better" if p_value < 0.05 else "nothing yet"
    print(f"  {n:>7}  {alo:.0%}-{ahi:.0%}         {blo2:.0%}-{bhi2:.0%}         "
          f"{'yes' if overlap else 'no':<10}p={p_value:<12.3f}{verdict}")

print("\nTwo things to take from that table.\n")
print("The first is how large n has to be. A three-point difference needs thousands of")
print("cases before a proper test will call it, and almost nobody has thousands of cases.")
print("That is not a reason to despair; it is a reason to stop arguing about three points.")
print()
print("The second is that 'do the intervals overlap' is much more conservative than the")
print("test. It says 'nothing yet' well past the point where the test says otherwise — so")
print("it is a safe instinct at a whiteboard and the wrong instrument for a decision.")

# ------------------------------------------------------------------ paired is much better
print("\n\nAnd there is a much cheaper way: run both systems on the SAME cases.\n")
cases = 120
both_right, both_wrong = 92, 12
a_only, b_only = 11, 5          # where they disagree — the only cases that carry signal
print(f"  both correct        {both_right:>4}")
print(f"  both wrong          {both_wrong:>4}")
print(f"  only A correct      {a_only:>4}   <- the signal")
print(f"  only B correct      {b_only:>4}   <- the signal")
discordant = a_only + b_only
print(f"\n  A scores {(both_right + a_only) / cases:.0%}, B scores "
      f"{(both_right + b_only) / cases:.0%} — a "
      f"{100 * (a_only - b_only) / cases:.0f}-point gap over {cases} cases.")
print(f"  But the comparison rests on only {discordant} cases, because the "
      f"{both_right + both_wrong} where they")
print("  agree carry no information about which is better.")

# McNemar's exact test, which is a binomial test on the discordant pairs.
p = sum(math.comb(discordant, k) for k in range(a_only, discordant + 1)) / 2 ** discordant
print(f"\n  McNemar's exact test on those {discordant}: p = {p:.3f} "
      f"({'significant' if p < 0.05 else 'not significant'} at 0.05)")
print("\nPairing removes the case-to-case variation that dominates an unpaired")
print("comparison, and it is free: you were going to run both systems anyway.")
