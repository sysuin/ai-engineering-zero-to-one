# When you want "only 2024 Q3", where the filter goes changes the answer.

import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED     # noqa: E402
from meridian_index import load_index      # noqa: E402

client = OpenAI()
chunks, vectors = load_index()

QUESTION = "What was revenue and which region was weakest?"
WANT = {"kind": "quarterly-review", "year": 2024, "quarter": 3}

q = np.array(client.embeddings.create(model=MODEL_EMBED, input=[QUESTION]).data[0]
             .embedding, dtype=np.float32)
q /= np.linalg.norm(q)
scores = vectors @ q


def matches(chunk: dict) -> bool:
    return all(chunk.get(k) == v for k, v in WANT.items())


K = 5

# Post-filter: search everything, then throw away what does not match.
top = np.argsort(-scores)[:K]
post = [chunks[i] for i in top if matches(chunks[i])]

# Pre-filter: restrict the candidates first, then search among them.
allowed = [i for i, c in enumerate(chunks) if matches(c)]
pre_order = sorted(allowed, key=lambda i: -scores[i])[:K]
pre = [chunks[i] for i in pre_order]

print(f"Question: {QUESTION}")
print(f"Filter:   {WANT}")
print(f"Corpus:   {len(chunks):,} chunks, {len(allowed)} of them match the filter\n")

print(f"Post-filter — search all {len(chunks):,}, keep the top {K} that match")
print(f"  returned {len(post)} of {K} wanted")
for c in post:
    print(f"    {c['source']:22} {c['heading'][:40]}")
if not post:
    print("    nothing: none of the top 5 overall happened to be from 2024 Q3")

print(f"\nPre-filter — restrict to the {len(allowed)} matching, then search")
print(f"  returned {len(pre)} of {K} wanted")
for c in pre:
    print(f"    {c['source']:22} {c['heading'][:40]}   {scores[chunks.index(c)]:.3f}")

Path("code/12/_filtering.json").write_text(json.dumps(
    {"corpus": len(chunks), "matching": len(allowed),
     "post_returned": len(post), "pre_returned": len(pre)}, indent=2))

print()
print("Post-filtering asks for the best five overall and hopes some of them qualify.")
print("When the filter is selective — here, 7 chunks out of 1,454 — most of the time")
print("none of them do, and you return nothing while the right answer sits in the index.")
print()
print("Pre-filtering asks for the best five among those that qualify. It always returns")
print("five if five exist. This is why 'does your vector database do filtered search")
print("properly?' is a real question and not a feature-list detail.")
