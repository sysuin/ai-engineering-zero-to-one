# The arithmetic of voting. Majority voting multiplies a model's accuracy only when its
# mistakes are independent. Here is how much, and what happens when they are not.

import math
import random


def majority_accuracy(p: float, n: int) -> float:
    """P(more than half of n independent answers are right), each right with probability p."""
    return sum(math.comb(n, k) * p**k * (1 - p)**(n - k) for k in range(n // 2 + 1, n + 1))


print("Independent errors: each sample right with probability p")
print(f"  {'p':>5} " + "".join(f"{f'vote of {n}':>11}" for n in (1, 3, 5, 9, 15)))
for p in (0.55, 0.75, 0.85):
    print(f"  {p:>5.2f} " + "".join(f"{majority_accuracy(p, n):>11.1%}" for n in (1, 3, 5, 9, 15)))

# Now a population of questions in which some are simply hard: on those, the model is wrong
# most of the time, whichever sample you take. The average single-sample accuracy is the same.
rng = random.Random(3)
QUESTIONS, HARD_SHARE = 20_000, 0.25
EASY_P, HARD_P = 0.9, 0.3             # 0.75 * 0.9 + 0.25 * 0.3 = 0.75 on average


def simulate(n: int) -> float:
    right = 0
    for _ in range(QUESTIONS):
        p = HARD_P if rng.random() < HARD_SHARE else EASY_P
        votes = sum(rng.random() < p for _ in range(n))
        right += votes > n / 2
    return right / QUESTIONS


print(f"\nCorrelated errors: {HARD_SHARE:.0%} of questions are hard (p={HARD_P}), "
      f"the rest easy (p={EASY_P})")
single = HARD_SHARE * HARD_P + (1 - HARD_SHARE) * EASY_P
print(f"  average single-sample accuracy {single:.1%}, the same as p=0.75 above")
for n in (1, 5, 15):
    print(f"  vote of {n:>2}: {simulate(n):.1%}   "
          f"(independent errors would give {majority_accuracy(single, n):.1%})")
limit = 1 - HARD_SHARE
print(f"\nas the vote grows, easy questions become always right and hard ones always wrong, so")
print(f"accuracy heads towards the share of easy questions, {limit:.0%}: voting cannot rescue a")
print("question the model usually gets wrong, and it makes that question wrong every time.")
