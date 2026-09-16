# skip
"""The pieces of Chapter 14's ablation that later listings reuse: the index, the test set's
truth, BM25, fusion and recall. Nothing here calls a model except `embed`."""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from _testset import QUESTIONS                          # noqa: E402
from clarity.config import MODEL_EMBED                  # noqa: E402
from meridian_index import load_index                   # noqa: E402

client = OpenAI()
chunks, vectors = load_index()


def answers(needle) -> set[int]:
    """Chunks that answer: any containing the needle, or the named section of a document, or
    any chunk matching one of a list of either."""
    if isinstance(needle, list):
        return set().union(*(answers(n) for n in needle))
    if isinstance(needle, tuple):
        source, heading = needle
        return {i for i, c in enumerate(chunks)
                if c["source"] == source and c.get("heading", "").startswith(heading)}
    return {i for i, c in enumerate(chunks) if needle in c["text"]}


TRUTH = [answers(needle) for _, needle, _ in QUESTIONS]


def embed(texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 256):
        out.extend(d.embedding for d in client.embeddings.create(
            model=MODEL_EMBED, input=texts[i:i + 256]).data)
    a = np.array(out, dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def tokenise(text): return re.findall(r"[a-z0-9\-]{2,}", text.lower())


DOCS = [tokenise(c["text"]) for c in chunks]
DF = Counter(w for d in DOCS for w in set(d))
AVG = sum(len(d) for d in DOCS) / len(DOCS)
COUNTS = [Counter(d) for d in DOCS]


def bm25_scores(query: str) -> np.ndarray:
    scores = np.zeros(len(DOCS))
    for term in tokenise(query):
        if term in DF:
            idf = math.log(1 + (len(DOCS) - DF[term] + 0.5) / (DF[term] + 0.5))
            for i, counts in enumerate(COUNTS):
                tf = counts.get(term, 0)
                if tf:
                    scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(DOCS[i]) / AVG))
    return scores


def bm25_order(query: str) -> list[int]:
    return list(np.argsort(-bm25_scores(query)))


def rrf(*rankings, k=1):
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            fused[i] += 1.0 / (k + rank)
    return sorted(fused, key=lambda i: -fused[i])


def recall_at(orders: list[list[int]], k: int = 5) -> float:
    return sum(bool(set(o[:k]) & t) for o, t in zip(orders, TRUTH)) / len(TRUTH)
