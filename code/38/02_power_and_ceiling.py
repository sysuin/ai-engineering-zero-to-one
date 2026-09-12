# How many cases do you need, and how high can you possibly score?
#
# Two questions that decide what your evaluation is capable of, both answerable before
# you write a single case.

import math
import random

random.seed(11)
Z95, Z80 = 1.959964, 0.841621     # 5% two-sided significance, 80% power


def cases_needed(baseline: float, lift: float,
                 z_alpha: float = Z95, z_beta: float = Z80) -> int:
    """
    Cases per arm to detect `lift` over `baseline`, paired-free and two-sided.

    The classic two-proportion sample-size formula. It is worth running once before any
    A/B argument, because the number is usually shocking and always deflationary.
    """
    p1, p2 = baseline, baseline + lift
    pbar = (p1 + p2) / 2
    numerator = (z_alpha * math.sqrt(2 * pbar * (1 - pbar))
                 + z_beta * math.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2
    return math.ceil(numerator / (lift ** 2))


print("Cases per arm needed to detect a real improvement, at 80% power.\n")
print(f"  {'baseline':>9}  " + "".join(f"{f'+{int(l * 100)}pt':>9}"
                                       for l in (0.01, 0.02, 0.03, 0.05, 0.10)))
for baseline in (0.60, 0.75, 0.85, 0.92):
    row = "".join(f"{cases_needed(baseline, lift):>9,}"
                  for lift in (0.01, 0.02, 0.03, 0.05, 0.10))
    print(f"  {baseline:>8.0%}  {row}")

print("\nRead the +3pt column, because that is the size of improvement people argue")
print("about most. It costs thousands of cases. Two consequences follow, and both are")
print("more useful than the despair:\n")
print("  1. Pair your comparisons. Running both systems on the same cases removes the")
print("     case-to-case variation and cuts the requirement dramatically — often by more")
print("     than half, and more when the systems agree often.")
print("  2. Stop shipping 3-point changes one at a time. Batch five of them, measure the")
print("     batch, and keep the batch or revert it. A 15-point difference is visible at")
print(f"     {cases_needed(0.85, 0.15):,} cases, which is a set you can actually build.")

# ------------------------------------------------------------------ the noise ceiling
print("\n\nThe noise ceiling: the highest score anybody could get on your set.\n")
print("Your labels are not perfect. §6.5 shows how an arguable ground truth caps every")
print("score you report, and a case whose gold answer is wrong is a case a perfect")
print("system fails.\n")
print(f"  {'label error':>12}  {'ceiling':>9}   what an 85% result actually means")
for error in (0.00, 0.03, 0.05, 0.10, 0.15):
    ceiling = 1 - error
    # A perfect system scores `ceiling`. A system scoring 0.85 has therefore closed
    # 0.85/ceiling of the achievable range.
    closed = min(0.85 / ceiling, 1.0)
    print(f"  {error:>11.0%}  {ceiling:>9.0%}   {closed:>4.0%} of what is achievable")

print("\nAt 10% label error a score of 85% is 94% of everything available, and the six")
print("points you are chasing are mostly wrong labels. Measuring the ceiling is how you")
print("find out that your model is finished and your data is not.")

# ------------------------------------------------------------------ run-to-run noise
print("\n\nAnd the other ceiling: the same system, scored twice.\n")
TRIALS, N, TRUE = 200, 120, 0.85
# Each case has some probability of passing; the system is unchanged between runs, so
# every point of spread below is the sampling of a non-deterministic system, not a change.
runs = [sum(random.random() < TRUE for _ in range(N)) / N for _ in range(TRIALS)]
runs.sort()
lo, hi = runs[int(0.025 * TRIALS)], runs[int(0.975 * TRIALS)]
print(f"  {TRIALS} runs of an unchanged system at a true rate of {TRUE:.0%}, "
      f"n={N} each:\n")
print(f"    lowest observed    {min(runs):.1%}")
print(f"    highest observed   {max(runs):.1%}")
print(f"    middle 95%         {lo:.1%} - {hi:.1%}   "
      f"({100 * (hi - lo):.0f} points wide)")
# "an 8-point gain", "a 7-point gain" — the numbers move between runs, so even the
# article has to be computed, like everything else in this book.
gain, drop = round(100 * (max(runs) - TRUE)), round(100 * (TRUE - min(runs)))
article = "an" if str(gain).startswith(("8", "11", "18")) else "a"
print("\n  Nothing changed between those runs. A team watching this dashboard would")
print(f"  have celebrated {article} {gain}-point gain and investigated a "
      f"{drop}-point regression,")
print("  and both would have been the same system on a different Tuesday.")
print("\n  This is the argument for §22.9's drift threshold against a recorded baseline,")
print("  rather than an absolute floor set just under today's score.")
