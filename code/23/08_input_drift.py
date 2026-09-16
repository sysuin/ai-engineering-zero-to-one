# timeout: 600
# Watching what users ask, not only how well it is answered. Two weeks of questions, a
# change in the mix between them, and a single number that says how far the mix moved.

import random
import re
import sys
from collections import Counter
from math import log

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                          # noqa: E402
from clarity.evals.runner import load                           # noqa: E402
from meridian_index import load_index                           # noqa: E402

random.seed(23)
cases = load()


def kind_of(question: str) -> str:
    """A cheap classifier, the kind a monitor can afford to run on every request."""
    q = question.lower()
    if re.search(r"msc-\d|contract|agreement|terminat|clause", q):
        return "contract"
    if re.search(r"revenue|margin|orders|units|sku|profit", q):
        return "figures"
    if re.search(r"depot|capacity|utilisation|storage", q):
        return "operations"
    return "other"


_, vectors = load_index()
unique = sorted({c["question"] for c in cases})
embedded = OpenAI().embeddings.create(model=MODEL_EMBED, input=unique).data
nearest = {q: float(np.max(vectors @ (np.array(e.embedding) / np.linalg.norm(e.embedding))))
           for q, e in zip(unique, embedded)}


def coverage_of(question: str) -> str:
    """How close the question is to anything in the corpus: one embedding, one dot product."""
    s = nearest[question]
    return "far" if s < 0.40 else "near" if s < 0.55 else "close"


def psi(before: Counter, after: Counter) -> float:
    """Population stability index: how far one distribution of categories moved."""
    total_b, total_a = sum(before.values()), sum(after.values())
    score = 0.0
    for key in set(before) | set(after):
        b = max(before[key] / total_b, 0.001)
        a = max(after[key] / total_a, 0.001)
        score += (a - b) * log(a / b)
    return score


answerable = [c["question"] for c in cases if c["kind"] != "unanswerable"]
outside = [c["question"] for c in cases if c["kind"] == "unanswerable"]
week_1 = random.choices(answerable, k=500) + random.choices(outside, k=25)
week_2 = random.choices(answerable, k=500) + random.choices(outside, k=25)
week_3 = random.choices(answerable, k=350) + random.choices(outside, k=175)

print(f"  population stability index      {'kind of question':>16} {'coverage':>9}")
for label, week in (("week 1 against week 2 (no change)", week_2),
                    ("week 1 against week 3 (the change)", week_3)):
    by_kind = psi(Counter(map(kind_of, week_1)), Counter(map(kind_of, week)))
    by_cover = psi(Counter(map(coverage_of, week_1)), Counter(map(coverage_of, week)))
    print(f"  {label:34} {by_kind:>13.3f} {by_cover:>9.3f}")

for name, feature in (("kind", kind_of), ("coverage", coverage_of)):
    first, third = Counter(map(feature, week_1)), Counter(map(feature, week_3))
    print(f"\n  share by {name:<21} week 1   week 3")
    for key in sorted(set(first) | set(third)):
        print(f"    {key:28} {first[key] / len(week_1):>6.0%} {third[key] / len(week_3):>8.0%}")
