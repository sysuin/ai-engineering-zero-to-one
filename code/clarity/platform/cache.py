"""
Clarity v0.20 — three layers of cache, and one of them is dangerous.

    exact       the same bytes; a dictionary keyed by a hash. Always safe.
    semantic    a similar question; a nearest neighbour over embeddings. Not safe.
    provider    the same prefix; the provider charges less for it. Safe and invisible.

The first and third are free wins with no correctness risk. The middle one trades
correctness for hit rate, and §28.4 measures the trade rather than asserting it is
fine.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_EMBED                          # noqa: E402


@dataclass
class ExactCache:
    """A hash of the exact request. No judgement, no risk, and a modest hit rate."""
    store: dict[str, str] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    @staticmethod
    def key(question: str, **context) -> str:
        payload = json.dumps({"q": question.strip().lower(), **context},
                             sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def get(self, question: str, **context) -> str | None:
        found = self.store.get(self.key(question, **context))
        self.hits += found is not None
        self.misses += found is None
        return found

    def put(self, question: str, answer: str, **context) -> None:
        self.store[self.key(question, **context)] = answer

    @property
    def rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0


class SemanticCache:
    """
    A nearest neighbour over question embeddings, above a similarity threshold.

    The threshold is the whole product. Set it low and the cache answers questions it
    was never asked; set it high and it never hits. There is no value that does both,
    and §28.4 shows why with two questions six words apart.
    """

    def __init__(self, client, threshold: float = 0.92) -> None:
        self.client = client
        self.threshold = threshold
        self.questions: list[str] = []
        self.answers: list[str] = []
        self.vectors: list[np.ndarray] = []
        self.hits = 0
        self.misses = 0

    def _embed(self, text: str) -> np.ndarray:
        vector = np.array(self.client.embeddings.create(
            model=MODEL_EMBED, input=[text]).data[0].embedding, dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def lookup(self, question: str) -> tuple[str | None, float, str | None]:
        if not self.vectors:
            self.misses += 1
            return None, 0.0, None
        scores = np.stack(self.vectors) @ self._embed(question)
        best = int(np.argmax(scores))
        if scores[best] >= self.threshold:
            self.hits += 1
            return self.answers[best], float(scores[best]), self.questions[best]
        self.misses += 1
        return None, float(scores[best]), self.questions[best]

    def put(self, question: str, answer: str) -> None:
        self.questions.append(question)
        self.answers.append(answer)
        self.vectors.append(self._embed(question))

    @property
    def rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
