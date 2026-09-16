"""
Clarity v0.6 — retrieval that earns its keep.

Measured in Chapter 14 on thirty questions (after pooling corrected seven of their answers),
three runs for anything that calls a model:

    dense only                77%
    + fused with BM25         80%      k=1 and k=60 alike on recall; k=1 slightly ahead on MRR
    + metadata filter      93-100%     extracted from the question, applied as a pre-filter
    + reranking            +0 to +2    a call per question — available, off by default
    + MMR                  +0          after the filter; no call, kept for varied top fives

Not yet done: a one-line header embedded with each chunk took dense search from 77% to 97%
with no model call. This file still searches the un-headed index of Chapters 11 to 13.

Known weakness: the facet extractor sometimes invents a year from a contract reference, the
combined filter then matches nothing, and the fallback drops the whole filter.
"""
from __future__ import annotations

import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_EMBED, MODEL_FAST      # noqa: E402


@dataclass
class Passage:
    text: str
    source: str
    heading: str
    score: float
    index: int


class Facets(BaseModel):
    """What a question restricts itself to, if anything."""
    year: int | None = Field(description="A year the question names, if any.")
    quarter: int | None = Field(description="A quarter 1-4, if the question names one.")
    contract_ref: str | None = Field(description="A reference like MSC-2022-100.")


class _Ranked(BaseModel):
    best_ids: list[int] = Field(description="Excerpt numbers, most relevant first.")


class Retriever:
    def __init__(self, chunks: list[dict], vectors: np.ndarray,
                 client: OpenAI | None = None):
        self.chunks = chunks
        self.vectors = vectors
        self.client = client or OpenAI()
        self._build_keyword_index()

    # ------------------------------------------------------------------ keyword half
    @staticmethod
    def _tokenise(text: str) -> list[str]:
        # Hyphens are kept, because MSC-2022-100 must survive as one term. Chapter 11
        # measured what embeddings do to identifiers; this is the half that fixes it.
        return re.findall(r"[a-z0-9\-]{2,}", text.lower())

    def _build_keyword_index(self) -> None:
        self._docs = [self._tokenise(c["text"]) for c in self.chunks]
        self._counts = [Counter(d) for d in self._docs]
        self._df = Counter(w for d in self._docs for w in set(d))
        self._avg_len = sum(len(d) for d in self._docs) / len(self._docs)

    def _bm25(self, query: str) -> np.ndarray:
        scores = np.zeros(len(self._docs))
        for term in self._tokenise(query):
            if term not in self._df:
                continue
            idf = math.log(1 + (len(self._docs) - self._df[term] + 0.5)
                           / (self._df[term] + 0.5))
            for i, counts in enumerate(self._counts):
                tf = counts.get(term, 0)
                if tf:
                    scores[i] += idf * tf * 2.5 / (
                        tf + 1.5 * (0.25 + 0.75 * len(self._docs[i]) / self._avg_len))
        return scores

    # ------------------------------------------------------------------ dense half
    def _embed(self, text: str) -> np.ndarray:
        vector = np.array(self.client.embeddings.create(
            model=MODEL_EMBED, input=[text]).data[0].embedding, dtype=np.float32)
        return vector / np.linalg.norm(vector)

    # ------------------------------------------------------------------ fusion
    @staticmethod
    def _fuse(*rankings: list[int], k: int = 1) -> list[int]:
        """
        Reciprocal rank fusion.

        `k` damps the advantage of being ranked first. A large k rewards chunks both
        retrievers found; a small k rewards one retriever's confident first place. On the
        corrected test set, 1 and 60 answered the same number of questions — each won one
        the other lost — and 1 ordered them slightly better. Measure it; do not inherit it.
        """
        fused: dict[int, float] = defaultdict(float)
        for ranking in rankings:
            for rank, i in enumerate(ranking, start=1):
                fused[i] += 1.0 / (k + rank)
        return sorted(fused, key=lambda i: -fused[i])

    # ------------------------------------------------------------------ facets
    def facets(self, question: str) -> Facets:
        parsed = self.client.chat.completions.parse(
            model=MODEL_FAST, temperature=0, max_completion_tokens=200,
            response_format=Facets,
            messages=[{"role": "system", "content":
                       "Extract only what the question explicitly names. Null "
                       "otherwise. Do not infer."},
                      {"role": "user", "content": question}],
        ).choices[0].message.parsed
        return parsed or Facets(year=None, quarter=None, contract_ref=None)

    def _allowed(self, facets: Facets) -> set[int] | None:
        if not any((facets.year, facets.quarter, facets.contract_ref)):
            return None
        keep = {i for i, c in enumerate(self.chunks)
                if (not facets.contract_ref
                    or facets.contract_ref in str(c.get("ref", "")))
                and (not facets.year or c.get("year") == facets.year)
                and (not facets.quarter or c.get("quarter") == facets.quarter)}
        # An empty result means the filter was wrong, not that the corpus is empty.
        return keep or None

    # ------------------------------------------------------------------ diversity
    def _mmr(self, order: list[int], query: np.ndarray, keep: int,
             diversity: float = 0.3) -> list[int]:
        chosen: list[int] = []
        pool = list(order[:30])
        while pool and len(chosen) < keep:
            best, best_score = pool[0], -1e9
            for i in pool:
                relevance = float(self.vectors[i] @ query)
                redundancy = max((float(self.vectors[i] @ self.vectors[j])
                                  for j in chosen), default=0.0)
                score = (1 - diversity) * relevance - diversity * redundancy
                if score > best_score:
                    best, best_score = i, score
            chosen.append(best)
            pool.remove(best)
        return chosen

    # ------------------------------------------------------------------ reranking
    def _rerank(self, question: str, candidates: list[int], keep: int) -> list[int]:
        listing = "\n\n".join(
            f"[{n}] {self.chunks[i]['source']} — {self.chunks[i].get('heading', '')}\n"
            f"{self.chunks[i]['text'][:400]}" for n, i in enumerate(candidates))
        parsed = self.client.chat.completions.parse(
            model=MODEL_FAST, temperature=0, max_completion_tokens=300,
            response_format=_Ranked,
            messages=[{"role": "system", "content":
                       "Order the excerpts by how well each answers the question."},
                      {"role": "user", "content":
                       f"<excerpts>\n{listing}\n</excerpts>\n\nQuestion: {question}"}],
        ).choices[0].message.parsed
        ordered = [candidates[n] for n in (parsed.best_ids if parsed else [])
                   if n < len(candidates)]
        return (ordered + [i for i in candidates if i not in ordered])[:keep]

    # ------------------------------------------------------------------ the pipeline
    def search(self, question: str, k: int = 5, *, use_facets: bool = True,
               use_rerank: bool = False) -> list[Passage]:
        query = self._embed(question)
        allowed = self._allowed(self.facets(question)) if use_facets else None

        dense_scores = self.vectors @ query
        pool = range(len(self.chunks)) if allowed is None else allowed
        dense = sorted(pool, key=lambda i: -dense_scores[i])[:50]

        keyword_scores = self._bm25(question)
        # Filter the keyword ranking BEFORE cutting it short. Taking the global top 200 and
        # filtering afterwards is a post-filter by another name: forty contracts share an
        # identically worded payment clause, and the one the filter wants can rank 201st.
        keyword = [i for i in np.argsort(-keyword_scores)
                   if allowed is None or i in allowed][:50]

        order = self._fuse(dense, keyword, k=1)
        if use_rerank:
            order = self._rerank(question, order[:25], 12)
        order = self._mmr(order, query, k)

        return [Passage(self.chunks[i]["text"], self.chunks[i]["source"],
                        self.chunks[i].get("heading", ""), float(dense_scores[i]), i)
                for i in order]
