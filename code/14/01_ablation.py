# timeout: 1800
# One technique at a time, measured on the same thirty questions.

import json
import math
import re
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, "code")
from _testset import QUESTIONS                          # noqa: E402
from clarity.config import MODEL_EMBED, MODEL_FAST      # noqa: E402
from meridian_index import load_index                   # noqa: E402

client = OpenAI()
chunks, vectors = load_index()
K = 5

# A question is answered if ANY chunk containing its needle is retrieved — several
# contracts answer "how long to pay", and any of them is correct.
TRUTH = [{i for i, c in enumerate(chunks) if needle in c["text"]}
         for _, needle, _ in QUESTIONS]


def embed(texts: list[str]) -> np.ndarray:
    out = []
    for i in range(0, len(texts), 128):
        out.extend(d.embedding for d in client.embeddings.create(
            model=MODEL_EMBED, input=texts[i:i + 128]).data)
    array = np.array(out, dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


QUERY_VECTORS = embed([q for q, _, _ in QUESTIONS])


# ---------------------------------------------------------------- BM25
def tokenise(text): return re.findall(r"[a-z0-9\-]{2,}", text.lower())


DOCS = [tokenise(c["text"]) for c in chunks]
DF = Counter(w for d in DOCS for w in set(d))
AVG = sum(len(d) for d in DOCS) / len(DOCS)
COUNTS = [Counter(d) for d in DOCS]


def bm25(query: str) -> np.ndarray:
    scores = np.zeros(len(DOCS))
    for term in tokenise(query):
        if term not in DF:
            continue
        idf = math.log(1 + (len(DOCS) - DF[term] + 0.5) / (DF[term] + 0.5))
        for i, counts in enumerate(COUNTS):
            tf = counts.get(term, 0)
            if tf:
                scores[i] += idf * tf * 2.5 / (tf + 1.5 * (0.25 + 0.75 * len(DOCS[i]) / AVG))
    return scores


def rrf(*rankings: list[int], k: int = 60) -> list[int]:
    """
    Reciprocal rank fusion: combine orderings without comparing their scores.

    `k` damps the advantage of being ranked first. The value 60 is quoted almost
    universally, and this chapter measures what it does here.
    """
    fused = defaultdict(float)
    for ranking in rankings:
        for rank, i in enumerate(ranking, start=1):
            fused[i] += 1.0 / (k + rank)
    return sorted(fused, key=lambda i: -fused[i])


# ---------------------------------------------------------------- metadata inference
class Facets(BaseModel):
    year: int | None = Field(description="A year the question restricts to, if any.")
    quarter: int | None = Field(description="A quarter 1-4, if the question names one.")
    contract_ref: str | None = Field(description="A contract reference like MSC-2022-100.")


def infer_facets(question: str) -> Facets:
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200,
        response_format=Facets,
        messages=[{"role": "system", "content":
                   "Extract only what the question explicitly names. Null otherwise."},
                  {"role": "user", "content": question}],
    ).choices[0].message.parsed
    return parsed or Facets(year=None, quarter=None, contract_ref=None)


def allowed_by(facets: Facets) -> set[int] | None:
    if not any((facets.year, facets.quarter, facets.contract_ref)):
        return None
    keep = set()
    for i, c in enumerate(chunks):
        if facets.contract_ref and facets.contract_ref not in c.get("ref", ""):
            continue
        if facets.year and c.get("year") != facets.year:
            continue
        if facets.quarter and c.get("quarter") != facets.quarter:
            continue
        keep.add(i)
    return keep or None


# ---------------------------------------------------------------- reranking
class Ranked(BaseModel):
    best_ids: list[int] = Field(description="Excerpt numbers, most relevant first.")


def rerank(question: str, candidates: list[int], keep: int) -> list[int]:
    listing = "\n\n".join(
        f"[{n}] {chunks[i]['source']} — {chunks[i].get('heading', '')}\n"
        f"{chunks[i]['text'][:400]}" for n, i in enumerate(candidates))
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        response_format=Ranked,
        messages=[{"role": "system", "content":
                   "Order the excerpts by how well each answers the question. Return "
                   "the numbers of the best ones, most relevant first."},
                  {"role": "user", "content":
                   f"<excerpts>\n{listing}\n</excerpts>\n\nQuestion: {question}"}],
    ).choices[0].message.parsed
    order = [candidates[n] for n in (parsed.best_ids if parsed else []) if n < len(candidates)]
    return (order + [i for i in candidates if i not in order])[:keep]


# ---------------------------------------------------------------- MMR
def mmr(order: list[int], query_vector: np.ndarray, keep: int,
        diversity: float = 0.3) -> list[int]:
    chosen: list[int] = []
    pool = list(order)
    while pool and len(chosen) < keep:
        best, best_score = None, -1e9
        for i in pool[:30]:
            relevance = float(vectors[i] @ query_vector)
            redundancy = max((float(vectors[i] @ vectors[j]) for j in chosen), default=0.0)
            score = (1 - diversity) * relevance - diversity * redundancy
            if score > best_score:
                best, best_score = i, score
        chosen.append(best)
        pool.remove(best)
    return chosen


# ---------------------------------------------------------------- the ablation
def dense_order(qi: int, allowed: set[int] | None = None) -> list[int]:
    scores = vectors @ QUERY_VECTORS[qi]
    pool = range(len(chunks)) if allowed is None else allowed
    return sorted(pool, key=lambda i: -scores[i])


FACETS = None


def run(name: str, fn) -> dict:
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(fn, range(len(QUESTIONS))))
    hits = [bool(set(r[:K]) & TRUTH[i]) for i, r in enumerate(results)]
    by_tag = defaultdict(lambda: [0, 0])
    for (q, _, tag), hit in zip(QUESTIONS, hits):
        by_tag[tag][0] += hit
        by_tag[tag][1] += 1
    return {"recall": sum(hits) / len(hits),
            "by_tag": {t: v[0] / v[1] for t, v in by_tag.items()},
            "hits": hits}


def keyword_order(i: int) -> list[int]:
    return list(np.argsort(-bm25(QUESTIONS[i][0])))


configs = {}
configs["dense only"] = run("dense", lambda i: dense_order(i))
configs["BM25 only"] = run("bm25", keyword_order)
configs["fused, k=60"] = run("rrf60", lambda i: rrf(
    dense_order(i)[:50], keyword_order(i)[:50]))
configs["fused, k=1"] = run("rrf1", lambda i: rrf(
    dense_order(i)[:50], keyword_order(i)[:50], k=1))

print("inferring metadata from each question …", file=sys.stderr)
with ThreadPoolExecutor(max_workers=10) as pool:
    FACETS = list(pool.map(lambda q: infer_facets(q[0]), QUESTIONS))


def hybrid_filtered(i: int) -> list[int]:
    allowed = allowed_by(FACETS[i])
    dense = dense_order(i, allowed)[:50]
    keyword = [j for j in keyword_order(i)[:200]
               if allowed is None or j in allowed][:50]
    return rrf(dense, keyword, k=1)


configs["+ metadata filter"] = run("filtered", hybrid_filtered)

configs["+ rerank top 25"] = run("reranked", lambda i: rerank(
    QUESTIONS[i][0], hybrid_filtered(i)[:25], K))

configs["+ MMR diversity"] = run("mmr", lambda i: mmr(
    rerank(QUESTIONS[i][0], hybrid_filtered(i)[:25], 12), QUERY_VECTORS[i], K))

TAGS = ["plain", "near-duplicate", "identifier", "paraphrase", "ticket", "obscure"]
print(f"\n{'configuration':21}{'r@5':>5}  " +
      "".join(f"{t[:7]:>7}" for t in TAGS))
for name, r in configs.items():
    row = "".join(f"{r['by_tag'].get(t, 0):>7.0%}" for t in TAGS)
    print(f"{name:21}{r['recall']:>5.0%}  {row}")

Path("code/14/_ablation.json").write_text(json.dumps(
    {"k": K, "n": len(QUESTIONS), "tags": TAGS,
     "configs": {k: {"recall": v["recall"], "by_tag": v["by_tag"]}
                 for k, v in configs.items()}}, indent=2))

first, last = list(configs.values())[0], list(configs.values())[-1]
print(f"\n{len(QUESTIONS)} questions, recall@{K}. "
      f"{first['recall']:.0%} to {last['recall']:.0%}.")
