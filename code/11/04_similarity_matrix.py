# Ten questions against the eight clauses of one contract, as a matrix of similarities — and
# the training objective that makes the right cell in each row the largest.

import math

import numpy as np
from openai import OpenAI

from _corpus import QUERIES, load
from clarity.config import MODEL_EMBED

client = OpenAI()
clauses = [c for c in load(limit=1)]
clauses.sort(key=lambda c: int(c["clause"]))


def embed(texts):
    data = client.embeddings.create(model=MODEL_EMBED, input=texts).data
    v = np.array([d.embedding for d in data])
    return v / np.linalg.norm(v, axis=1, keepdims=True)


Q = embed([q for q, _ in QUERIES])
C = embed([c["text"] for c in clauses])
S = Q @ C.T                                        # 10 questions x 8 clauses

print("similarity, question (row) against clause (column) of", clauses[0]["contract"], "\n")
print("        " + "".join(f"{'cl ' + c['clause']:>7}" for c in clauses) + "   want")
right = 0
for i, (question, want) in enumerate(QUERIES):
    best = int(np.argmax(S[i]))
    right += clauses[best]["clause"] == want
    cells = "".join(f"{S[i, j]:>6.2f}{'*' if j == best else ' '}" for j in range(len(clauses)))
    print(f"  q{i + 1:<3} {cells}   cl {want}")
print(f"\n* the best clause in each row; it is the right one for {right} of {len(QUERIES)}")


def info_nce(tau: float) -> float:
    """Average -log softmax probability of the right clause, at temperature tau."""
    total = 0.0
    for i, (_, want) in enumerate(QUERIES):
        logits = S[i] / tau
        logits -= logits.max()
        j = next(k for k, c in enumerate(clauses) if c["clause"] == want)
        total -= logits[j] - math.log(np.exp(logits).sum())
    return total / len(QUERIES)


print("\ncontrastive loss on these ten rows (lower means the right clause stands out more)")
for tau in (1.0, 0.1, 0.05, 0.02):
    print(f"  temperature {tau:<5} loss {info_nce(tau):6.3f}")
print(f"  a model that knew nothing would score log({len(clauses)}) = {math.log(len(clauses)):.3f}")
