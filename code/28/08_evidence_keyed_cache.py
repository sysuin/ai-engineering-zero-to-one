# timeout: 900
# "Key the cache on the retrieved evidence, not on the question." The same pairs as the semantic
# cache listing, keyed three ways. Evidence keys are only as good as the retrieval that makes them.

import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                   # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from meridian_index import load_index                    # noqa: E402

client = OpenAI()
chunks, vectors = load_index()
retriever = Retriever(chunks, vectors)

DANGEROUS = [                                            # must not share an answer
    ("What was revenue in 2024 Q3?", "What was revenue in 2024 Q4?"),
    ("What was revenue in 2024 Q3?", "What was gross profit in 2024 Q3?"),
    ("Which region was strongest in 2024 Q3?", "Which region was weakest in 2024 Q3?"),
    ("What is the price cap under MSC-2022-100?", "What is the price cap under MSC-2024-118?"),
    ("How many orders were there in 2025 Q1?", "How many units were shipped in 2025 Q1?"),
    ("What was Midwest revenue in 2024 Q3?", "What was Midwest revenue in 2025 Q3?"),
]
SAFE = [                                                 # should share one
    ("What was revenue in 2024 Q3?", "what was the revenue for 2024 q3"),
    ("Which region was weakest in 2024 Q3?", "In 2024 Q3, which region performed worst?"),
    ("What are the payment terms under MSC-2022-100?", "Under MSC-2022-100, how long do we have to pay?"),
]


def embed(text: str) -> np.ndarray:
    v = np.array(client.embeddings.create(model=MODEL_EMBED, input=[text]).data[0].embedding)
    return v / np.linalg.norm(v)


def keys(question: str) -> dict:
    v = embed(question)
    return {"vector": v,
            "plain top 5": frozenset(int(i) for i in np.argsort(-(vectors @ v))[:5]),
            "v0.6 ranked": [p.index for p in retriever.search(question, k=5)]}


def shares(a: dict, b: dict) -> dict:
    return {"question vectors ≥ 0.92": float(a["vector"] @ b["vector"]) >= 0.92,
            "same plain top 5": a["plain top 5"] == b["plain top 5"],
            "same v0.6 top 5": set(a["v0.6 ranked"]) == set(b["v0.6 ranked"]),
            "same v0.6 first passage": a["v0.6 ranked"][:1] == b["v0.6 ranked"][:1]}


KEYS = ["question vectors ≥ 0.92", "same plain top 5", "same v0.6 top 5", "same v0.6 first passage"]
cache = {}
SHORT = dict(zip(KEYS, ["question ≥0.92", "plain top 5", "v0.6 top 5", "v0.6 first"]))
print(f"  {'second question of the pair':44}" + "".join(f"{SHORT[k]:>16}" for k in KEYS))
totals = {k: {"dangerous": 0, "safe": 0} for k in KEYS}
for label, pairs in (("dangerous", DANGEROUS), ("safe", SAFE)):
    print(f"  -- {'must not share' if label == 'dangerous' else 'should share'}")
    for first, second in pairs:
        for q in (first, second):
            cache.setdefault(q, keys(q))
        result = shares(cache[first], cache[second])
        for k in KEYS:
            totals[k][label] += result[k]
        print(f"  {second[:44]:44}" + "".join(f"{'shared' if result[k] else '·':>16}" for k in KEYS))

print()
for k in KEYS:
    print(f"  {k:26} served a wrong answer on {totals[k]['dangerous']} of {len(DANGEROUS)} pairs, "
          f"hit on {totals[k]['safe']} of {len(SAFE)} rephrasings")
