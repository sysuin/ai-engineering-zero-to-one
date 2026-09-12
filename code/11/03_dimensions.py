# timeout: 900
# How many numbers do you actually need?

import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from _corpus import QUERIES, load
from clarity.config import MODEL_EMBED

client = OpenAI()
corpus = load()


def embed(texts: list[str], dimensions: int | None = None) -> np.ndarray:
    kwargs = {"dimensions": dimensions} if dimensions else {}
    vectors = []
    for i in range(0, len(texts), 128):
        vectors.extend(d.embedding for d in client.embeddings.create(
            model=MODEL_EMBED, input=texts[i:i + 128], **kwargs).data)
    array = np.array(vectors)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


TEXTS = [c["text"] for c in corpus]
QUESTIONS = [q for q, _ in QUERIES]
WANT = [w for _, w in QUERIES]

full = embed(TEXTS)
print(f"The model's native size is {full.shape[1]} numbers per chunk.\n")
print(f"{'dimensions':>11} {'recall@1':>10} {'recall@3':>10} "
      f"{'index size':>13} {'of full':>9}")

results = {}
for dims in (64, 128, 256, 512, 1024, None):
    vectors = embed(TEXTS, dims) if dims else full
    queries = embed(QUESTIONS, dims) if dims else embed(QUESTIONS)
    n = vectors.shape[1]

    hits1 = hits3 = 0
    for qv, want in zip(queries, WANT):
        order = np.argsort(-(vectors @ qv))
        hits1 += corpus[order[0]]["clause"] == want
        hits3 += any(corpus[i]["clause"] == want for i in order[:3])

    # Four bytes per number, which is what a float32 index actually costs.
    size = len(corpus) * n * 4
    results[n] = {"recall_1": hits1 / len(WANT), "recall_3": hits3 / len(WANT),
                  "bytes": size}
    print(f"{n:>11} {hits1 / len(WANT):>9.0%} {hits3 / len(WANT):>10.0%} "
          f"{size / 1024:>11.1f} KB {size / (len(corpus) * full.shape[1] * 4):>8.0%}")

Path("code/11/_dimensions.json").write_text(json.dumps(results, indent=2))

print()
print("These vectors are Matryoshka embeddings: the model is trained so that the first")
print("N numbers are a usable embedding on their own. Truncating is not lossy in the")
print("way that throwing away half a normal vector would be — the important information")
print("is deliberately packed into the front.")
print()
print("Which means the size of your index is a dial, not a property of the model.")
print("Extrapolate, because this is where the decision actually lands:\n")
print(f"  {'dimensions':>11} {'96 chunks':>12} {'1M chunks':>12} {'100M chunks':>14}")
for n in sorted(results):
    per_chunk = n * 4
    print(f"  {n:>11} {per_chunk * len(corpus) / 1024:>9.0f} KB "
          f"{per_chunk * 1_000_000 / 1e9:>9.1f} GB "
          f"{per_chunk * 100_000_000 / 1e9:>11.0f} GB")
big, small = max(results), 512
ratio = max(results) / small
print(f"\nAt a hundred million chunks, {big} numbers is "
      f"{big * 4 * 100_000_000 / 1e9:.0f} GB and {small} is "
      f"{small * 4 * 100_000_000 / 1e9:.0f} GB —")
print(f"a factor of {ratio:.0f}, and the difference between two quite different pieces")
print("of hardware. Measure the recall you lose before you pay for dimensions you")
print("do not need.")
