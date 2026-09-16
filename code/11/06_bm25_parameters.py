# BM25, taken apart: what its two parameters do, measured on the same ten questions.

import math
import re
from collections import Counter

from _corpus import QUERIES, load

corpus = load()


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


DOCS = [tokenise(c["text"]) for c in corpus]
DF = Counter(w for d in DOCS for w in set(d))
AVG = sum(len(d) for d in DOCS) / len(DOCS)


def bm25(query, k1, b):
    out = []
    for doc in DOCS:
        counts, score = Counter(doc), 0.0
        for t in tokenise(query):
            if t in counts:
                idf = math.log(1 + (len(DOCS) - DF[t] + 0.5) / (DF[t] + 0.5))
                tf = counts[t]
                score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / AVG))
        out.append(score)
    return out


def recall_at(k1, b, k):
    hits = 0
    for q, want in QUERIES:
        scores = bm25(q, k1, b)
        top = sorted(range(len(corpus)), key=lambda i: -scores[i])[:k]
        hits += any(corpus[i]["clause"] == want for i in top)
    return hits / len(QUERIES)


print("term frequency saturation: the contribution of a word seen tf times (k1 = 1.5, average length)")
print("  tf      " + "".join(f"{tf:>6}" for tf in (1, 2, 3, 5, 10, 50)))
print("  weight  " + "".join(f"{tf * 2.5 / (tf + 1.5):>6.2f}" for tf in (1, 2, 3, 5, 10, 50)))

print("\nidf: rare words count for more")
for word in ("the", "supplier", "invoice", "delaware"):
    df = DF.get(word, 0)
    idf = math.log(1 + (len(DOCS) - df + 0.5) / (df + 0.5))
    print(f"  {word:10} in {df:>3} of {len(DOCS)} clauses   idf {idf:.2f}")

print(f"\nrecall@1 / recall@3 on the ten questions")
print("          " + "".join(f"{'b=' + str(b):>12}" for b in (0.0, 0.75, 1.0)))
for k1 in (0.0, 0.5, 1.5, 3.0):
    print(f"  k1={k1:<4} " + "".join(f"{recall_at(k1, b, 1):>7.0%}/{recall_at(k1, b, 3):<4.0%}"
                                     for b in (0.0, 0.75, 1.0)))
