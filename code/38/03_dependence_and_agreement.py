# Three ways an evaluation's arithmetic goes wrong without any arithmetic mistake:
# cases that are not independent, labellers who agree by chance, and too many slices.

import math

import numpy as np

rng = np.random.default_rng(38)
Z = 1.959964


def wilson(successes, n):
    p = successes / n
    d = 1 + Z * Z / n
    centre = (p + Z * Z / (2 * n)) / d
    half = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return centre - half, centre + half


# ------------------------------------------------------------------ 1. clustered cases
TEMPLATES, PER_TEMPLATE, TRUE_RATE, SPREAD = 40, 3, 0.85, 2.0
SETS, RESAMPLES = 1_000, 400
# Each template has its own pass probability, drawn around the true rate: cases from one
# template pass or fail together. SPREAD is the Beta distribution's a + b; smaller is lumpier.
a, b = TRUE_RATE * SPREAD, (1 - TRUE_RATE) * SPREAD
rates = rng.beta(a, b, (SETS, TEMPLATES))
passed = rng.binomial(PER_TEMPLATE, rates)                       # per template, per set
n = TEMPLATES * PER_TEMPLATE
truth = TRUE_RATE              # the rate over every template that could have been written
lo, hi = wilson(passed.sum(axis=1), n)
naive = ((lo <= truth) & (truth <= hi)).mean()

pick = rng.integers(0, TEMPLATES, (SETS, RESAMPLES, TEMPLATES))  # resample templates
boot = np.take_along_axis(passed[:, None, :].repeat(RESAMPLES, 1), pick, 2).sum(2) / n
blo, bhi = np.percentile(boot, 2.5, axis=1), np.percentile(boot, 97.5, axis=1)
clustered = ((blo <= truth) & (truth <= bhi)).mean()

icc = 1 / (SPREAD + 1)
deff = 1 + (PER_TEMPLATE - 1) * icc
print(f"1. {n} cases from {TEMPLATES} templates of {PER_TEMPLATE}, "
      f"intra-template correlation {icc:.2f}\n")
print(f"  design effect 1 + (m - 1) x ICC       {deff:.2f}")
print(f"  effective number of cases              {n / deff:.0f} of {n}")
print(f"  Wilson interval, treating cases as independent   covers the truth "
      f"{naive:.0%}")
print(f"  bootstrap that resamples templates               covers the truth "
      f"{clustered:.0%}")
print("  (a 95% interval should cover it 95% of the time)")

# ------------------------------------------------------------------ 2. agreement
print("\n2. Two labellers, 200 answers\n")


def kappa(both_pass, a_only, b_only, both_fail):
    total = both_pass + a_only + b_only + both_fail
    observed = (both_pass + both_fail) / total
    a_pass = (both_pass + a_only) / total
    b_pass = (both_pass + b_only) / total
    chance = a_pass * b_pass + (1 - a_pass) * (1 - b_pass)
    return observed, chance, (observed - chance) / (1 - chance)


print(f"  {'':<34} {'agree':>6} {'by chance':>10} {'kappa':>6}")
for label, table in (("balanced: half the answers pass", (90, 10, 10, 90)),
                     ("skewed: nearly every answer passes", (180, 10, 10, 0))):
    observed, chance, k = kappa(*table)
    print(f"  {label:<34} {observed:>6.0%} {chance:>10.0%} {k:>6.2f}")

# ------------------------------------------------------------------ 3. many slices
SLICES, TRIALS = 20, 5_000
print(f"\n3. {SLICES} slices, no real difference in any of them\n")
p = rng.uniform(size=(TRIALS, SLICES))                          # null p-values
any_raw = (p < 0.05).any(axis=1).mean()
any_bonferroni = (p < 0.05 / SLICES).any(axis=1).mean()
ranked = np.sort(p, axis=1)
holm = (ranked < 0.05 / (SLICES - np.arange(SLICES))).cumprod(axis=1).any(axis=1).mean()
print(f"  at least one slice 'significant' at p<0.05    {any_raw:.0%}")
print(f"  ...with Bonferroni (p < 0.05 / {SLICES})           {any_bonferroni:.0%}")
print(f"  ...with Holm's step-down version              {holm:.0%}")
print(f"\n  Expected by arithmetic: 1 - 0.95^{SLICES} = {1 - 0.95 ** SLICES:.0%}.")
