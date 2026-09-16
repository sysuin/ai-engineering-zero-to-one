# Pseudo-relevance feedback: assume the first results are right, learn from them, search again.
# Rocchio's update for dense vectors and term expansion for BM25, with no model call.

import math
from collections import Counter

import numpy as np

from _retrieval import DF, DOCS, QUESTIONS, TRUTH, bm25_order, embed, recall_at, tokenise, vectors

Q = embed([q for q, _, _ in QUESTIONS])
TAGS = ("plain", "near-duplicate", "identifier", "paraphrase", "ticket", "obscure")


def rocchio(q: np.ndarray, top: int, beta: float) -> np.ndarray:
    first = np.argsort(-(vectors @ q))[:top]
    moved = q + beta * vectors[first].mean(axis=0)
    return moved / np.linalg.norm(moved)


def expanded(question: str, top: int, terms: int) -> tuple[str, list[str]]:
    """Add the most distinctive words of the first results; repeat the original words to keep
    them weighted above the additions."""
    first = bm25_order(question)[:top]
    asked = set(tokenise(question))
    weight = Counter()
    for i in first:
        for word, count in Counter(DOCS[i]).items():
            if word not in asked and DF[word] > 1:
                weight[word] += count * math.log(len(DOCS) / DF[word])
    added = [w for w, _ in weight.most_common(terms)]
    return " ".join([question, question] + added), added


def row(name, orders):
    cells = ""
    for tag in TAGS:
        idx = [i for i, (_, _, t) in enumerate(QUESTIONS) if t == tag]
        cells += f"{sum(bool(set(orders[i][:5]) & TRUTH[i]) for i in idx) / len(idx):>9.0%}"
    print(f"  {name:31}{recall_at(orders):>5.0%}{cells}")


print(f"  {'recall@5':31}{'all':>5}" + "".join(f"{t[:8]:>9}" for t in TAGS))
row("dense", [np.argsort(-(vectors @ q)) for q in Q])
for top, beta in [(3, 0.5), (3, 1.0), (10, 1.0)]:
    row(f"  + Rocchio, top {top}, beta {beta}",
        [np.argsort(-(vectors @ rocchio(q, top, beta))) for q in Q])

row("BM25", [bm25_order(question) for question, _, _ in QUESTIONS])
for top, terms in [(3, 5), (3, 10), (10, 10)]:
    row(f"  + expansion, top {top}, {terms} words",
        [bm25_order(expanded(question, top, terms)[0]) for question, _, _ in QUESTIONS])

for n in (18, 12):
    question = QUESTIONS[n][0]
    print(f"\n{question!r}\n   BM25 expansion added: {', '.join(expanded(question, 3, 10)[1])}")
