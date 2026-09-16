# Recall@k, MRR and nDCG, computed on the same retrieval — and a worked nDCG example, because
# the formula is easier to trust once you have done it by hand.

import math

import numpy as np

from _retrieval import QUESTIONS, TRUTH, bm25_order, chunks, embed, recall_at, rrf, vectors

Q = embed([q for q, _, _ in QUESTIONS])
orders = [rrf([int(i) for i in np.argsort(-(vectors @ q))[:50]], bm25_order(question)[:50])
          for q, (question, _, _) in zip(Q, QUESTIONS)]


def first_hit(order, truth):
    return next((rank for rank, i in enumerate(order, start=1) if i in truth), None)


mrr = sum(1 / r if (r := first_hit(o, t)) else 0 for o, t in zip(orders, TRUTH)) / len(TRUTH)
print("fused dense + BM25 (k=1), no filter, 30 questions")
print(f"  recall@1 {recall_at(orders, 1):.0%}   recall@5 {recall_at(orders, 5):.0%}   "
      f"recall@10 {recall_at(orders, 10):.0%}   MRR {mrr:.3f}")

ranks = [first_hit(o, t) for o, t in zip(orders, TRUTH)]
print("  rank of the first right chunk, per question:",
      " ".join(str(r) if r and r <= 20 else "-" for r in ranks))


def dcg(relevances):
    return sum(rel / math.log2(position + 1) for position, rel in enumerate(relevances, start=1))


# Graded relevance for one question: 2 = answers it, 1 = relevant but does not answer, 0 = neither.
system_a = [2, 0, 1, 0, 0]
system_b = [0, 1, 2, 0, 0]
ideal = sorted(system_a, reverse=True)
print("\nworked nDCG@5, one question, grades 2 / 1 / 0")
for name, grades in [("system A", system_a), ("system B", system_b), ("ideal", ideal)]:
    print(f"  {name:9} {grades}   DCG {dcg(grades):.3f}   nDCG {dcg(grades) / dcg(ideal):.3f}")
print("  both systems found the same two useful chunks; nDCG rewards the one that put the")
print("  better chunk first, which recall@5 cannot see")
