# timeout: 600
# Depends on the Meridian index (meridian_index.py) and the golden set.
# Storing each number of an embedding in fewer bits: what it saves, and what it costs in
# finding the right chunk — measured on real questions, against the full vectors.

import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                          # noqa: E402
from clarity.evals.runner import load, plain                     # noqa: E402
from meridian_index import load_index                            # noqa: E402

K, SHORTLIST = 5, 50
chunks, vectors = load_index()
texts = [plain(" ".join(c["text"].split())) for c in chunks]

cases = []
for case in load():
    if not case.get("span"):
        continue
    probe = plain(" ".join(case["span"].split())[:60])
    gold = {i for i, t in enumerate(texts) if probe in t}
    if gold:
        cases.append((case["question"], gold))

client = OpenAI()
queries = np.array([d.embedding for d in client.embeddings.create(
    model=MODEL_EMBED, input=[q for q, _ in cases]).data], dtype=np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

# int8: one scale per dimension, from the corpus's own range.
low, high = vectors.min(axis=0), vectors.max(axis=0)
scale = (high - low) / 255
as_int8 = np.round((vectors - low) / scale).astype(np.uint8)
q_int8 = np.clip(np.round((queries - low) / scale), 0, 255).astype(np.uint8)
# binary: one bit per dimension, the sign.
bits = np.packbits(vectors > 0, axis=1)
q_bits = np.packbits(queries > 0, axis=1)


def top(scores: np.ndarray, k: int) -> np.ndarray:
    return np.argsort(-scores, axis=1)[:, :k]


def recall(ranked: np.ndarray) -> float:
    return np.mean([bool(set(r) & gold) for r, (_, gold) in zip(ranked, cases)])


exact = top(queries @ vectors.T, K)
dequant = (as_int8.astype(np.float32) * scale + low)
int8_rank = top((q_int8.astype(np.float32) * scale + low) @ dequant.T, K)
hamming = np.unpackbits(q_bits[:, None, :] ^ bits[None, :, :], axis=2).sum(axis=2)
binary_rank = np.argsort(hamming, axis=1)[:, :K]
shortlist = np.argsort(hamming, axis=1)[:, :SHORTLIST]
rescored = np.array([s[np.argsort(-(vectors[s] @ q))][:K] for s, q in zip(shortlist, queries)])

agree = lambda ranked: np.mean([len(set(a) & set(b)) / K for a, b in zip(ranked, exact)])  # noqa: E731
print(f"{len(cases)} golden questions, {len(chunks):,} chunks, recall@{K} of the verified chunk\n")
print(f"  {'vectors stored as':<34} {'bytes each':>10} {'recall@5':>9} {'same top 5':>11}")
for label, size, ranked in (("32-bit floats (full)", vectors.shape[1] * 4, exact),
                            ("8-bit integers", vectors.shape[1], int8_rank),
                            ("1 bit per dimension", bits.shape[1], binary_rank),
                            (f"1 bit, then rescore top {SHORTLIST} in full", bits.shape[1],
                             rescored)):
    print(f"  {label:<34} {size:>10,} {recall(ranked):>9.0%} {agree(ranked):>11.0%}")

if recall(binary_rank) >= recall(exact):
    print(f"\nRecall alone hides what one bit did: it held at {recall(binary_rank):.0%} on "
          f"{len(cases)} questions while only {agree(binary_rank):.0%} of")
    print("each top five stayed the same. The right chunk survived, its neighbours changed, and")
    print("whether that costs recall is a question for a much larger set than this one.")
print(f"\nA bit per dimension is {vectors.shape[1] * 4 // bits.shape[1]} times smaller than the "
      "full vector. On its own it reorders the")
print("top results; used as a first pass, with the shortlist rescored against full vectors kept")
print("on disk, it gives back the full ranking for a fraction of the memory.")
