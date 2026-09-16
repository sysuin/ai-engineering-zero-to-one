# A few points on thirty questions: a difference, or one or two questions that happened to
# change? Pairs of configurations, compared question by question.

import math
import random

import numpy as np

from _retrieval import QUESTIONS, TRUTH, bm25_order, embed, rrf, vectors

Q = embed([q for q, _, _ in QUESTIONS])
DENSE = [list(np.argsort(-(vectors @ q))) for q in Q]
KEYWORD = [bm25_order(question) for question, _, _ in QUESTIONS]


def hits(orders):
    return [bool(set(o[:5]) & t) for o, t in zip(orders, TRUTH)]


H = {"dense": hits(DENSE),
     "BM25": hits(KEYWORD),
     "fused k=60": hits([rrf(d[:50], k[:50], k=60) for d, k in zip(DENSE, KEYWORD)]),
     "fused k=1": hits([rrf(d[:50], k[:50], k=1) for d, k in zip(DENSE, KEYWORD)])}


def mcnemar_exact(fixed: int, broken: int) -> float:
    """Two-sided exact test: of the questions that changed, is the split a coin toss?"""
    n = fixed + broken
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, j) for j in range(min(fixed, broken) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def bootstrap_interval(a, b, draws=10_000, seed=0):
    rng = random.Random(seed)
    n, diffs = len(a), []
    for _ in range(draws):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    return diffs[int(0.025 * draws)], diffs[int(0.975 * draws)]


print(f"{'from':12}{'to':12}{'change':>8}{'fixed':>7}{'broke':>7}{'exact p':>9}   95% interval")
for a, b in [("dense", "fused k=1"), ("fused k=60", "fused k=1"), ("dense", "fused k=60"),
             ("BM25", "dense")]:
    fixed = sum(1 for x, y in zip(H[a], H[b]) if y and not x)
    broken = sum(1 for x, y in zip(H[a], H[b]) if x and not y)
    low, high = bootstrap_interval(H[a], H[b])
    change = 100 * (sum(H[b]) - sum(H[a])) / len(QUESTIONS)
    print(f"{a:12}{b:12}{change:>+7.0f}pt{fixed:>7}{broken:>7}{mcnemar_exact(fixed, broken):>9.3f}"
          f"   {100 * low:+.0f} to {100 * high:+.0f} points")


CRITICAL = {}


def critical(d: int, alpha: float) -> int:
    """Fewest fixes among d changed questions that the exact test calls a gain."""
    if d not in CRITICAL:
        CRITICAL[d] = next((f for f in range(d // 2 + 1, d + 1) if mcnemar_exact(f, d - f) < alpha), d + 1)
    return CRITICAL[d]


def binomial(n: int, j: int, p: float) -> float:
    if p <= 0 or p >= 1:
        return float(j == (n if p >= 1 else 0))
    return math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                    + j * math.log(p) + (n - j) * math.log(1 - p))


def power(n: int, p_fix: float, p_break: float, alpha=0.05) -> float:
    """Chance an exact paired test on n questions detects a gain with these rates."""
    p_change, share = p_fix + p_break, p_fix / (p_fix + p_break)
    total = 0.0
    for d in range(n + 1):
        p_d = binomial(n, d, p_change)
        if p_d > 1e-12:
            total += p_d * sum(binomial(d, f, share) for f in range(critical(d, alpha), d + 1))
    return total


def questions_needed(p_fix, p_break, target=0.8):
    return next(n for n in range(10, 5000, 5) if power(n, p_fix, p_break) >= target)


print("\nquestions a paired test needs to see a gain 80% of the time, at the 5% level")
print(f"  {'gain':>6}  {'questions it fixes':>19}  {'questions it breaks':>20}  {'needed':>7}")
for p_fix, p_break in [(0.07, 0.0), (0.10, 0.03), (0.15, 0.08), (0.20, 0.10), (0.25, 0.05)]:
    print(f"  {100 * (p_fix - p_break):>+5.0f}pt  {p_fix:>19.0%}  {p_break:>20.0%}  "
          f"{questions_needed(p_fix, p_break):>7}")
print(f"\npower of these 30 questions for the first row: {power(30, 0.07, 0.0):.0%}")
