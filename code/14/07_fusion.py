# Fusion, more carefully: the constant swept, scores fused instead of ranks, and how deep each
# list is read. No model call except embedding the thirty questions.

import json
from pathlib import Path

import numpy as np

from _retrieval import QUESTIONS, TRUTH, bm25_scores, embed, recall_at, rrf, vectors

Q = embed([q for q, _, _ in QUESTIONS])
DENSE_SCORES = [vectors @ q for q in Q]
KEYWORD_SCORES = [bm25_scores(question) for question, _, _ in QUESTIONS]
DENSE = [np.argsort(-s) for s in DENSE_SCORES]
KEYWORD = [np.argsort(-s) for s in KEYWORD_SCORES]


def mrr(orders) -> float:
    total = 0.0
    for order, truth in zip(orders, TRUTH):
        rank = next((r for r, i in enumerate(order, start=1) if i in truth), None)
        total += 1 / rank if rank else 0
    return total / len(orders)


def show(name, orders):
    print(f"  {name:34} recall@5 {recall_at(orders):>4.0%}   MRR {mrr(orders):.3f}")
    return {"recall": recall_at(orders), "mrr": mrr(orders)}


def rank_fusion(k: int, depth: int = 50):
    return [rrf(d[:depth].tolist(), w[:depth].tolist(), k=k) for d, w in zip(DENSE, KEYWORD)]


def rescaled(scores: np.ndarray, top: np.ndarray, how: str) -> dict[int, float]:
    s = scores[top]
    if how == "min-max":
        span = s.max() - s.min()
        s = (s - s.min()) / span if span else np.ones_like(s)
    else:                                                     # z-score
        s = (s - s.mean()) / (s.std() or 1.0)
    return dict(zip(top.tolist(), s.tolist()))


def score_fusion(qi: int, weight_dense: float, how: str = "min-max", mnz: bool = False):
    dense = rescaled(DENSE_SCORES[qi], DENSE[qi][:50], how)
    keyword = rescaled(KEYWORD_SCORES[qi], KEYWORD[qi][:50], how)
    floor_d, floor_w = min(dense.values()), min(keyword.values())
    fused = {}
    for i in dense.keys() | keyword.keys():
        # A chunk missing from one list gets that list's lowest rescaled score.
        fused[i] = weight_dense * dense.get(i, floor_d) + (1 - weight_dense) * keyword.get(i, floor_w)
        if mnz:                                  # CombMNZ: reward being found by both
            fused[i] *= (i in dense) + (i in keyword)
    return sorted(fused, key=lambda i: -fused[i])


print("reciprocal rank fusion, top 50 of each list")
by_k = {k: show(f"k = {k}", rank_fusion(k)) for k in (1, 2, 5, 10, 20, 60, 200)}
print(f"  what rank 1 is worth against rank 20:  k=1 {21 / 2:.1f}x   k=10 {30 / 11:.1f}x   "
      f"k=60 {80 / 61:.2f}x")

print("\nfusing the scores instead of the ranks, top 50 of each list")
by_weight = {w: show(f"min-max, dense weight {w:.1f}", [score_fusion(i, w) for i in range(len(Q))])
             for w in (0.0, 0.3, 0.5, 0.7, 1.0)}
show("z-score, dense weight 0.5", [score_fusion(i, 0.5, "z-score") for i in range(len(Q))])
show("min-max, weight 0.5, CombMNZ", [score_fusion(i, 0.5, mnz=True) for i in range(len(Q))])



def rank_in(order, truth):
    return next((r for r, i in enumerate(order, start=1) if i in truth), None)


print("\nquestions in the top 5 at one constant and not the other, with the rank in each list")
for a, b in ((1, 60), (60, 1)):
    for qi, (question, _, _) in enumerate(QUESTIONS):
        at_a, at_b = rank_fusion(a)[qi], rank_fusion(b)[qi]
        if set(at_a[:5]) & TRUTH[qi] and not set(at_b[:5]) & TRUTH[qi]:
            print(f"  only at k = {a:<3} {question[:40]:41} dense {rank_in(DENSE[qi], TRUTH[qi])}, "
                  f"BM25 {rank_in(KEYWORD[qi], TRUTH[qi])}")

print("\nhow deep each list is read before fusing (RRF, k = 1)")
for depth in (5, 10, 50, 200):
    show(f"top {depth} of each", rank_fusion(1, depth))

Path("code/14/_fusion.json").write_text(json.dumps({"k": {str(k): v for k, v in by_k.items()}},
                                                   indent=2))
recalls = [v["recall"] for v in by_k.values()]
low, high = min(by_k, key=lambda k: by_k[k]["mrr"]), max(by_k, key=lambda k: by_k[k]["mrr"])
spread = (f"was {recalls[0]:.0%} at every k" if min(recalls) == max(recalls)
          else f"ranged from {min(recalls):.0%} to {max(recalls):.0%}")
print(f"\nacross k, rank fusion's recall@5 {spread}; "
      f"MRR ranged from {by_k[low]['mrr']:.3f} (k = {low}) to {by_k[high]['mrr']:.3f} (k = {high})")
best_w = max(by_weight, key=lambda w: by_weight[w]["recall"])
print(f"score fusion's best min-max weight was {best_w:.1f}, at {by_weight[best_w]['recall']:.0%}")
