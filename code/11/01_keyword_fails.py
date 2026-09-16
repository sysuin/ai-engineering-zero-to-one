# timeout: 900
# Where matching words stops working.

import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np
from openai import OpenAI

from _corpus import QUERIES, load
from clarity.config import MODEL_EMBED

client = OpenAI()
corpus = load()
print(f"{len(corpus)} clauses from "
      f"{len({c['contract'] for c in corpus})} contracts\n")


# ---------------------------------------------------------------- keyword search
def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


DOCS = [tokenise(c["text"]) for c in corpus]
DF = Counter(word for doc in DOCS for word in set(doc))
AVG_LEN = sum(len(d) for d in DOCS) / len(DOCS)


def bm25(query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Standard keyword ranking. Rewards rare words, discounts long documents."""
    terms = tokenise(query)
    scores = []
    for doc in DOCS:
        counts = Counter(doc)
        score = 0.0
        for term in terms:
            if term not in counts:
                continue
            idf = math.log(1 + (len(DOCS) - DF[term] + 0.5) / (DF[term] + 0.5))
            tf = counts[term]
            score += idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * len(doc) / AVG_LEN))
        scores.append(score)
    return scores


# ---------------------------------------------------------------- semantic search
def embed(texts: list[str]) -> np.ndarray:
    vectors = []
    for i in range(0, len(texts), 128):
        response = client.embeddings.create(model=MODEL_EMBED, input=texts[i:i + 128])
        vectors.extend(item.embedding for item in response.data)
    array = np.array(vectors)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


VECTORS = embed([c["text"] for c in corpus])
QUERY_VECTORS = embed([q for q, _ in QUERIES])

print(f"{'question':44}{'BM25':>10}{'semantic':>10}{'':>10}")
keyword_hits = semantic_hits = 0
rows = []
for (question, want), qv in zip(QUERIES, QUERY_VECTORS):
    kw_top = corpus[int(np.argmax(bm25(question)))]
    sem_top = corpus[int(np.argmax(VECTORS @ qv))]
    kw_ok, sem_ok = kw_top["clause"] == want, sem_top["clause"] == want
    keyword_hits += kw_ok
    semantic_hits += sem_ok
    rows.append({"question": question, "want": want,
                 "bm25": kw_top["clause"], "semantic": sem_top["clause"]})
    flags = ("" if kw_ok else " kw") + ("" if sem_ok else " sem")
    print(f"{question[:43]:44}{('cl ' + kw_top['clause']):>10}"
          f"{('cl ' + sem_top['clause']):>10}{flags:>10}")

n = len(QUERIES)
print(f"\n{'BM25 (word overlap)':28} {keyword_hits}/{n}  {keyword_hits / n:.0%}")
print(f"{'embeddings (meaning)':28} {semantic_hits}/{n}  {semantic_hits / n:.0%}")

Path("code/11/_search.json").write_text(json.dumps(
    {"rows": rows, "bm25": keyword_hits / n, "semantic": semantic_hits / n}, indent=2))

print()
print("The questions were written the way a person asks them, not the way a contract is")
print("drafted. Where BM25 missed and meaning did not:")
for row in rows:
    if row["bm25"] != row["want"] and row["semantic"] == row["want"]:
        right = next(c for c in corpus if c["clause"] == row["want"])
        shared = sorted(set(tokenise(row["question"])) & set(tokenise(right["text"])))
        print(f"  {row['question']!r}")
        print(f"     words shared with the right clause: {', '.join(shared) or 'none'}")
