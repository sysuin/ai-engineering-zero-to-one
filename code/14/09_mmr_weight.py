# The diversity weight in maximal marginal relevance, swept from none to nearly all, on the
# unfiltered hybrid's top 25. What it buys in variety, and what it costs in recall.

from itertools import combinations

import numpy as np

from _retrieval import QUESTIONS, TRUTH, bm25_order, chunks, embed, recall_at, rrf, vectors

Q = embed([q for q, _, _ in QUESTIONS])
POOLS = [rrf([int(i) for i in np.argsort(-(vectors @ q))[:50]], bm25_order(question)[:50], k=1)[:25]
         for q, (question, _, _) in zip(Q, QUESTIONS)]


def mmr(pool: list[int], query: np.ndarray, keep: int, diversity: float) -> list[int]:
    chosen, rest = [], list(pool)
    while rest and len(chosen) < keep:
        def score(i):
            redundancy = max((float(vectors[i] @ vectors[j]) for j in chosen), default=0.0)
            return (1 - diversity) * float(vectors[i] @ query) - diversity * redundancy
        best = max(rest, key=score)
        chosen.append(best)
        rest.remove(best)
    return chosen


def variety(orders):
    sources = np.mean([len({chunks[i]["source"] for i in o[:5]}) for o in orders])
    similarity = np.mean([np.mean([float(vectors[a] @ vectors[b]) for a, b in combinations(o[:5], 2)])
                          for o in orders])
    return sources, similarity


def by_tag(orders, tag):
    idx = [i for i, (_, _, t) in enumerate(QUESTIONS) if t == tag]
    return sum(bool(set(orders[i][:5]) & TRUTH[i]) for i in idx) / len(idx)


TAGS = ("identifier", "paraphrase", "near-duplicate")
print(f"  {'top 5 chosen by':24}{'r@5':>5}" + "".join(f"{t[:10]:>11}" for t in TAGS)
      + f"{'sources':>9}{'similarity':>11}")


def row(name, orders):
    sources, similarity = variety(orders)
    print(f"  {name:24}{recall_at(orders):>5.0%}" + "".join(f"{by_tag(orders, t):>11.0%}" for t in TAGS)
          + f"{sources:>9.1f}{similarity:>11.2f}")
    return recall_at(orders)


row("fused order", POOLS)
results = {d: row(f"MMR, diversity {d:.1f}", [mmr(p, q, 5, d) for p, q in zip(POOLS, Q)])
           for d in (0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9)}

best = max(results.values())
kept = [d for d, r in results.items() if r == best]
print(f"\nbest recall@5 {best:.0%}, at diversity {', '.join(f'{d:.1f}' for d in kept)}; "
      f"at 0.9 it was {results[0.9]:.0%}")
print("sources = distinct documents in the top 5; similarity = mean cosine between its pairs")
