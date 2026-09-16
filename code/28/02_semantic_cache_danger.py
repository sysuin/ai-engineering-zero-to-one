# timeout: 1800
# The cache layer that trades correctness for hit rate, measured.

import json
import sys

import numpy as np

sys.path.insert(0, "code")
from clarity.evals.runner import load                          # noqa: E402
from clarity.platform.cache import ExactCache, SemanticCache   # noqa: E402
from openai import OpenAI                                      # noqa: E402

client = OpenAI()

# Pairs that are close in embedding space and must not share an answer. Every one is
# a question a real user asks, and the second is not a rephrasing of the first.
DANGEROUS = [
    ("What was revenue in 2024 Q3?", "What was revenue in 2024 Q4?"),
    ("What was revenue in 2024 Q3?", "What was gross profit in 2024 Q3?"),
    ("Which region was strongest in 2024 Q3?",
     "Which region was weakest in 2024 Q3?"),
    ("What is the price cap under MSC-2022-100?",
     "What is the price cap under MSC-2024-118?"),
    ("How many orders were there in 2025 Q1?",
     "How many units were shipped in 2025 Q1?"),
    ("What was Midwest revenue in 2024 Q3?",
     "What was Midwest revenue in 2025 Q3?"),
]
# And pairs that mean the same thing and should share one.
SAFE = [
    ("What was revenue in 2024 Q3?", "what was the revenue for 2024 q3"),
    ("Which region was weakest in 2024 Q3?",
     "In 2024 Q3, which region performed worst?"),
    ("What are the payment terms under MSC-2022-100?",
     "Under MSC-2022-100, how long do we have to pay?"),
]

cache = SemanticCache(client, threshold=0.92)


def similarity(a: str, b: str) -> float:
    return float(cache._embed(a) @ cache._embed(b))


print("Pairs of questions, and how similar their embeddings are.\n")
print(f"  {'':<52}{'similarity':>12}")
rows = {"dangerous": [], "safe": []}
for first, second in DANGEROUS:
    score = similarity(first, second)
    rows["dangerous"].append({"a": first, "b": second, "score": score})
    print(f"  {second[:50]:<52}{score:>12.3f}")
print()
for first, second in SAFE:
    score = similarity(first, second)
    rows["safe"].append({"a": first, "b": second, "score": score})
    print(f"  {second[:50]:<52}{score:>12.3f}")

dangerous = [r["score"] for r in rows["dangerous"]]
safe = [r["score"] for r in rows["safe"]]
print()
print(f"  must NOT share an answer: {min(dangerous):.3f} to {max(dangerous):.3f}")
print(f"  should share an answer:   {min(safe):.3f} to {max(safe):.3f}")

overlap = max(dangerous) >= min(safe)
best = None
for threshold in [t / 1000 for t in range(700, 1000)]:
    wrong = sum(1 for s in dangerous if s >= threshold)
    missed = sum(1 for s in safe if s < threshold)
    if best is None or wrong + missed < best[1]:
        best = (threshold, wrong + missed, wrong, missed)
json.dump({**rows, "overlap": overlap, "best_threshold": best[0],
           "best_errors": best[1]}, open("code/28/_semantic.json", "w"), indent=1)

print()
if overlap:
    print(f"  the two ranges overlap: no threshold separates them")
    print(f"  the best possible cut is {best[0]:.3f}, and it still gets "
          f"{best[1]} of {len(dangerous) + len(safe)} pairs wrong")
    print(f"    ({best[2]} dangerous pairs served from cache, {best[3]} safe pairs "
          f"missed)")
else:
    print(f"  the ranges are separable here, at {best[0]:.3f} — on nine pairs, which "
          f"is not")
    print("  a number to set a production threshold from")

print()
print("That is the failure mode, and it is not a tuning problem.")
print()
print("'Revenue in 2024 Q3' and 'revenue in 2024 Q4' are one character apart and mean")
print("entirely different things. An embedding is a measure of *aboutness*, and two")
print("questions about the same thing with different answers are exactly what it is")
print("built to score as similar.")
print()
print("So a semantic cache does not have a safe threshold. It has a threshold you")
print("have chosen a wrongness rate for, and the honest version of shipping one is:")
print()
print("  measure the rate on your own question pairs, not on a benchmark")
print("  never cache anything containing a figure, a date or an identifier")
print("  key on the *retrieved documents* rather than the question, so two questions")
print("    only share an answer when they share their evidence")
print("  and expire it fast, because a wrong cached answer is wrong every time")
print()
print("An exact cache has none of these problems, hits less often, and is where to")
print("start. §28.5 measures what it is worth on a mix of repeated questions.")
