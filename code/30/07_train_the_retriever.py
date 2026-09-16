# timeout: 900
# Training the retriever instead of the generator: a low-rank adapter on the query embedding,
# learned from the golden set's questions and the passages that answer them. Measured twice —
# on a random split, and on a split that holds out whole kinds of question.

import random
import re
import sys

import numpy as np
import yaml
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                  # noqa: E402
from meridian_index import load_index                   # noqa: E402

client = OpenAI()
chunks, vectors = load_index()                           # frozen: the index is never re-embedded
norm = lambda s: " ".join(s.split())                     # noqa: E731

cases = [c for c in yaml.safe_load(open("code/clarity/evals/golden.yaml"))["cases"]
         if c["kind"] == "document" and c.get("span")]
truth = [{i for i, ch in enumerate(chunks) if norm(c["span"]) in norm(ch["text"])} for c in cases]
kind = [re.sub(r"\d{4}-Q\d-", "", c["id"]) for c in cases]        # qbr-2024-Q3-revenue -> qbr-revenue
Q = np.array([d.embedding for d in client.embeddings.create(
    model=MODEL_EMBED, input=[c["question"] for c in cases]).data], dtype=np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)


def recall_at_5(queries: np.ndarray, rows: list[int]) -> float:
    return np.mean([bool(set(np.argsort(-(vectors @ q))[:5].tolist()) & truth[i]) for q, i in zip(queries, rows)])


def train_adapter(rows: list[int], rank: int = 8, epochs: int = 200, lr: float = 0.02, tau: float = 0.05):
    """q' = q + B(Aq): LoRA's shape, on the query side only. Softmax over every chunk in the index."""
    rng = np.random.default_rng(30)
    d = Q.shape[1]
    A = rng.normal(0, 0.01, (rank, d)).astype(np.float32)
    B = np.zeros((d, rank), dtype=np.float32)
    X = Q[rows]
    targets = [min(truth[i]) for i in rows]
    for _ in range(epochs):
        H = X @ A.T                                              # (n, r)
        P = X + H @ B.T                                          # adapted queries
        logits = (P @ vectors.T) / tau
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        probs /= probs.sum(axis=1, keepdims=True)
        probs[np.arange(len(rows)), targets] -= 1.0              # d loss / d logits
        G = (probs @ vectors) / (tau * len(rows))                # d loss / d adapted query
        grad_B = G.T @ H
        grad_A = (G @ B).T @ X
        B -= lr * grad_B
        A -= lr * grad_A
    return lambda queries: queries + (queries @ A.T) @ B.T


def cross_validate(folds: list[list[int]]) -> None:
    before = after = on_train = 0.0
    cells = []
    for hold in folds:
        train = [i for i in range(len(cases)) if i not in hold]
        adapt = train_adapter(train)
        b, a = recall_at_5(Q[hold], hold), recall_at_5(adapt(Q[hold]), hold)
        before, after = before + b * len(hold), after + a * len(hold)
        on_train += recall_at_5(adapt(Q[train]), train) / len(folds)
        cells.append(f"{b:.0%}->{a:.0%}")
    n = sum(len(f) for f in folds)
    print(f"    each fold's holdout, before -> after: {'  '.join(cells)}")
    print(f"    all holdout questions: {before / n:.0%} -> {after / n:.0%};   training questions after: {on_train:.0%}")


rows = list(range(len(cases)))
shuffled = rows[:]
random.Random(30).shuffle(shuffled)
tails = [i for i in rows if kind[i].startswith("tail")]
quarterly = sorted({k for k in kind if k.startswith("qbr")})

print(f"{len(cases)} golden-set questions answered by a passage; {len(quarterly)} quarterly kinds x 12 quarters, "
      f"{len(tails)} one-off questions")
print(f"recall@5 with the plain query embedding: {recall_at_5(Q, rows):.0%}\n")
print("five folds, rows assigned at random:")
cross_validate([shuffled[k::5] for k in range(5)])
print("\nfive folds, each holding out one quarterly kind of question entirely (and a share of the one-offs):")
cross_validate([[i for i in rows if kind[i] == k] + tails[f::5] for f, k in enumerate(quarterly)])
print(f"\nthe adapter trains {2 * 8 * Q.shape[1]:,} numbers; the index's {len(chunks):,} vectors are untouched")


def header(c: dict) -> str:                              # Chapter 14's one-line header, no training at all
    parts = [c["source"].removesuffix(".md")]
    if c.get("year"):
        parts.append(f"{c['year']} Q{c['quarter']}")
    if c.get("ref"):
        parts.append(f"agreement {c['ref']}")
    parts.append(c.get("heading", ""))
    return " · ".join(p for p in parts if p)


texts = [f"{header(c)}\n\n{c['text']}" for c in chunks]
headed = np.array([d.embedding for i in range(0, len(texts), 256) for d in client.embeddings.create(
    model=MODEL_EMBED, input=texts[i:i + 256]).data], dtype=np.float32)
headed /= np.linalg.norm(headed, axis=1, keepdims=True)
with_headers = np.mean([bool(set(np.argsort(-(headed @ q))[:5].tolist()) & truth[i]) for i, q in enumerate(Q)])
print(f"for comparison, the rung below: the same {len(cases)} questions against an index re-embedded with "
      f"Chapter 14's headers, no training: {with_headers:.0%}")
