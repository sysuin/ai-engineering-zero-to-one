# Pooling: collect what several retrievers returned, judge the chunks the test set did not
# expect, and correct the truth. This is what the corrections changed.

import numpy as np

from _retrieval import QUESTIONS, TRUTH, answers, bm25_order, embed, recall_at, rrf, vectors

# The needles of the first version, for the questions pooling corrected.
FIRST_VERSION = {
    8: "Revenue for 2025 Q4 was",
    10: "2023 Q3",
    11: "Revenue for 2025 Q2 was",
    18: "days of the date of a valid invoice",
    19: "terminate this Agreement for convenience",
    21: "aggregate liability shall not exceed",
    22: "order lines complete and on time",
}
FIRST = [answers(FIRST_VERSION[i]) if i in FIRST_VERSION else truth
         for i, truth in enumerate(TRUTH)]

Q = embed([q for q, _, _ in QUESTIONS])
DENSE = [[int(i) for i in np.argsort(-(vectors @ q))] for q in Q]
KEYWORD = [bm25_order(question) for question, _, _ in QUESTIONS]
CONFIGS = {"dense only": DENSE, "BM25 only": KEYWORD,
           "fused, k=60": [rrf(d[:50], k[:50], k=60) for d, k in zip(DENSE, KEYWORD)],
           "fused, k=1": [rrf(d[:50], k[:50], k=1) for d, k in zip(DENSE, KEYWORD)]}

pool = [set().union(*(set(orders[i][:5]) for orders in CONFIGS.values()))
        for i in range(len(QUESTIONS))]
unexpected = sum(len(p - FIRST[i]) for i, p in enumerate(pool))
print(f"pool: the top 5 of four retrievers, {sum(map(len, pool))} chunks over 30 questions")
print(f"  chunks the first truth did not count, each to be judged: {unexpected}")
print(f"  questions whose truth was corrected: {len(FIRST_VERSION)}")
for i, needle in FIRST_VERSION.items():
    gained, lost = len(TRUTH[i] - FIRST[i]), len(FIRST[i] - TRUTH[i])
    print(f"    {QUESTIONS[i][0][:46]:47} {len(FIRST[i]):>3} -> {len(TRUTH[i]):>3} chunks"
          f"  (+{gained} -{lost})")

TAGS = ["near-duplicate", "paraphrase"]
print(f"\n  {'recall@5':14}{'first truth':>13}{'corrected':>11}"
      + "".join(f"{t[:10]:>13}{'':>2}" for t in TAGS))


def tag_recall(orders, truth, tag):
    idx = [i for i, (_, _, t) in enumerate(QUESTIONS) if t == tag]
    return sum(bool(set(orders[i][:5]) & truth[i]) for i in idx) / len(idx)


for name, orders in CONFIGS.items():
    first = sum(bool(set(o[:5]) & t) for o, t in zip(orders, FIRST)) / len(QUESTIONS)
    cells = "".join(f"{tag_recall(orders, FIRST, t):>7.0%} ->{tag_recall(orders, TRUTH, t):>4.0%}"
                    for t in TAGS)
    print(f"  {name:14}{first:>13.0%}{recall_at(orders):>11.0%}{cells}")
