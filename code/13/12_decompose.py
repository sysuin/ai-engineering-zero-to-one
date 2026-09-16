# timeout: 600
# A two-part question retrieves for its stronger half. Split it first, retrieve per part, and
# count whether both halves' answers reach the prompt — at the same number of excerpts.

import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED, MODEL_FAST  # noqa: E402
from meridian_index import load_index               # noqa: E402

client = OpenAI()
chunks, vectors = load_index()

# (question, what answers the first part, what answers the second); a tuple is a heading prefix
# in any contract, because every contract answers these.
TWO_PART = [
    ("How many days notice before a price rise, and how long to pay an invoice?",
     ("2. Prices",), ("3. Payment",)),
    ("Which account did not renew, and what is the racked storage capacity at Dallas?",
     "Halloway Group did not renew", "Racked storage"),
    ("When did the Sanitation category launch, and what caused the spike in support contacts?",
     "Sanitation category launched this quarter", "batch defect in cloths"),
    ("Which law governs the supplier agreements, and how long can the buyer take to reject goods?",
     "governed by the laws", "reject non-conforming goods"),
    ("What happens at the Columbus depot in December, and what was received short there?",
     "close for inventory count", "12 short"),
    ("What is the purchase order approval threshold, and what insurance must a new supplier hold?",
     "require a second approval", "product liability"),
]


class Parts(BaseModel):
    questions: list[str]


def split(question: str) -> list[str]:
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200, response_format=Parts,
        messages=[{"role": "system", "content": "Split the question into the separate standalone "
                   "questions it asks. If it asks only one thing, return it unchanged."},
                  {"role": "user", "content": question}]).choices[0].message.parsed
    return parsed.questions if parsed and parsed.questions else [question]


def embed(texts):
    a = np.array([d.embedding for d in client.embeddings.create(model=MODEL_EMBED, input=texts).data],
                 dtype=np.float32)
    return a / np.linalg.norm(a, axis=1, keepdims=True)


def found(ids, needle) -> bool:
    if isinstance(needle, tuple):
        return any(chunks[i]["kind"] == "contract" and chunks[i]["heading"].startswith(needle[0])
                   for i in ids)
    return any(needle in chunks[i]["text"] for i in ids)


def top(vector, k):
    return [int(i) for i in np.argsort(-(vectors @ vector))[:k]]


with ThreadPoolExecutor(max_workers=6) as pool:
    parts = list(pool.map(split, [q for q, _, _ in TWO_PART]))

def rank(vector, needle) -> int:
    return next(r for r, i in enumerate(np.argsort(-(vectors @ vector)), start=1)
                if found([int(i)], needle))


whole = embed([q for q, _, _ in TWO_PART])
rows = {"one retrieval, 6 excerpts": [], "split, 3 excerpts per part": [],
        "whole and parts, 2 each": [], "split, 6 per part": []}
ranks = []
for (question, first, second), v, subs in zip(TWO_PART, whole, parts):
    sub_vectors = embed(subs)
    rows["one retrieval, 6 excerpts"].append(top(v, 6))
    rows["split, 3 excerpts per part"].append(list(dict.fromkeys(i for sv in sub_vectors for i in top(sv, 3))))
    rows["whole and parts, 2 each"].append(list(dict.fromkeys(
        i for sv in [v, *sub_vectors] for i in top(sv, 2))))
    rows["split, 6 per part"].append(list(dict.fromkeys(i for sv in sub_vectors for i in top(sv, 6))))
    paired = len(subs) == 2
    ranks.append((question, rank(v, first), rank(v, second),
                  rank(sub_vectors[0], first) if paired else None,
                  rank(sub_vectors[1], second) if paired else None, subs))

print(f"{len(TWO_PART)} two-part questions; the model split them into "
      f"{', '.join(str(len(p)) for p in parts)} parts\n")
print("rank of each half's answer: searched with the whole question -> with its own part")
for question, w1, w2, p1, p2, subs in ranks:
    print(f"  {question[:58]:59} {w1:>4} -> {p1 if p1 else '-':<4} {w2:>4} -> {p2 if p2 else '-'}")

print(f"\n  {'both halves in the prompt':28}{'first':>7}{'second':>8}{'both':>6}{'excerpts':>10}")
for name, retrieved in rows.items():
    a = [found(ids, f) for ids, (_, f, _) in zip(retrieved, TWO_PART)]
    b = [found(ids, s) for ids, (_, _, s) in zip(retrieved, TWO_PART)]
    both = sum(x and y for x, y in zip(a, b))
    print(f"  {name:28}{sum(a):>7}{sum(b):>8}{both:>6}{np.mean([len(r) for r in retrieved]):>10.1f}")

print("\nhow the model split two of them:")
for i in (1, 4):
    print(f"  {TWO_PART[i][0]!r}\n    -> {parts[i]}")
