"""
Clarity v0.5 — the whole retrieval pipeline, end to end.

Two paths, and keeping them separate is most of what makes this maintainable:

    index()    once per document change:  load -> clean -> chunk -> embed -> store
    answer()   once per question:         embed -> retrieve -> ground -> generate

The design decision that matters most is in `answer()`: it does not use the similarity
score to decide whether it knows something. Chapter 13 measured that a threshold gets
62% of that judgement right, and reading the retrieved text gets 83% with no false
confidence at all.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_EMBED, MODEL_FAST     # noqa: E402
from clarity.prompts import load as load_prompt        # noqa: E402


@dataclass
class Passage:
    """A retrieved chunk, and everything needed to cite it."""
    text: str
    source: str
    heading: str
    score: float


@dataclass
class Answer:
    """An answer, or an honest account of why there is not one."""
    text: str
    passages: list[Passage] = field(default_factory=list)
    grounded: bool = True
    missing: str | None = None
    prompt_tokens: int = 0

    @property
    def sources(self) -> list[str]:
        return sorted({p.source for p in self.passages})


class _Grounding(BaseModel):
    answer_is_present: bool = Field(
        description="True only if the excerpts contain enough to answer directly. "
                    "Being about the same subject is not enough.")
    what_is_missing: str | None = Field(
        description="If not present, the specific thing the excerpts lack.")


class Clarity:
    def __init__(self, chunks: list[dict], vectors: np.ndarray,
                 client: OpenAI | None = None):
        self.chunks = chunks
        self.vectors = vectors
        self.client = client or OpenAI()

    # ------------------------------------------------------------------ retrieval
    def _embed(self, text: str) -> np.ndarray:
        vector = np.array(self.client.embeddings.create(
            model=MODEL_EMBED, input=[text]).data[0].embedding, dtype=np.float32)
        return vector / np.linalg.norm(vector)

    def retrieve(self, question: str, k: int = 6,
                 where: dict | None = None) -> list[Passage]:
        scores = self.vectors @ self._embed(question)
        candidates = range(len(self.chunks))
        if where:
            # Pre-filter, never post-filter. Chapter 12 measured the difference.
            candidates = [i for i in candidates
                          if all(self.chunks[i].get(f) == v for f, v in where.items())]
        best = sorted(candidates, key=lambda i: -scores[i])[:k]
        return [Passage(self.chunks[i]["text"], self.chunks[i]["source"],
                        self.chunks[i].get("heading", ""), float(scores[i]))
                for i in best]

    # ------------------------------------------------------------------ grounding
    def _grounded(self, question: str, passages: list[Passage]) -> _Grounding:
        context = self._context(passages)
        parsed = self.client.chat.completions.parse(
            model=MODEL_FAST, temperature=0, max_completion_tokens=400,
            response_format=_Grounding,
            messages=[{"role": "system", "content":
                       "Decide whether the excerpts answer the question. Being about "
                       "the same subject is not enough."},
                      {"role": "user", "content":
                       f"<excerpts>\n{context}\n</excerpts>\n\nQuestion: {question}"}],
        ).choices[0].message.parsed
        return parsed or _Grounding(answer_is_present=False,
                                    what_is_missing="could not be determined")

    @staticmethod
    def _context(passages: list[Passage]) -> str:
        # The source label goes with each excerpt so the model can cite it, and so a
        # reader can check. Ordering is by relevance, best first.
        return "\n\n".join(f"[{p.source}] {p.heading}\n{p.text}" for p in passages)

    # ------------------------------------------------------------------ the answer
    def answer(self, question: str, k: int = 6,
               where: dict | None = None) -> Answer:
        passages = self.retrieve(question, k=k, where=where)
        if not passages:
            return Answer("Nothing in the corpus matched that.", grounded=False,
                          missing="no passages matched the filter")

        check = self._grounded(question, passages)
        if not check.answer_is_present:
            return Answer(
                f"The documents I have do not answer that. Missing: "
                f"{check.what_is_missing}",
                passages=passages, grounded=False, missing=check.what_is_missing)

        response = self.client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=500,
            messages=[{"role": "system", "content": load_prompt("answer")},
                      {"role": "user", "content":
                       f"<excerpts>\n{self._context(passages)}\n</excerpts>\n\n"
                       f"Question: {question}"}],
        )
        return Answer((response.choices[0].message.content or "").strip(),
                      passages=passages,
                      prompt_tokens=response.usage.prompt_tokens)
