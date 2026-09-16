# timeout: 900
# Clarity's index was built with MODEL_EMBED. What happens when a query vector comes from a
# different model — the same length, so nothing complains — and what a half-finished migration
# to another model does to recall. The other models are whatever this account offers.

import math
import re
import statistics
import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                  # noqa: E402
from clarity.evals.runner import load                   # noqa: E402
from meridian_index import load_index                   # noqa: E402

client = OpenAI()
chunks, vectors = load_index()
DIMS = vectors.shape[1]
cases = [c for c in load() if c["kind"] == "document" and c["source"]]
questions = [c["question"] for c in cases]


def embed(model: str, texts: list[str], dimensions: int | None = None) -> np.ndarray:
    extra = {"dimensions": dimensions} if dimensions else {}
    rows = []
    for start in range(0, len(texts), 256):
        reply = client.embeddings.create(model=model, input=texts[start:start + 256], **extra)
        rows += [d.embedding for d in reply.data]
    matrix = np.array(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def same_shape_as_the_index(model: str) -> tuple[str, int | None] | None:
    """A model whose vectors fit the index without complaint, natively or when asked."""
    for dimensions in (None, DIMS):
        try:
            if embed(model, ["test"], dimensions).shape[1] == DIMS:
                return model, dimensions
        except Exception:                                # noqa: BLE001 — not offered
            continue
    return None


def search(index: np.ndarray, queries: np.ndarray, k: int = 5):
    scores = queries @ index.T                           # raises if the lengths differ
    top = np.argsort(-scores, axis=1)[:, :k]
    found = [any(chunks[i]["source"] == c["source"] for i in row) for row, c in zip(top, cases)]
    return sum(found), statistics.median(scores.max(axis=1))


others = [m.id for m in client.models.list() if "embedding" in m.id and m.id != MODEL_EMBED]
fits = [f for f in map(same_shape_as_the_index, sorted(others)) if f]

print(f"index: {len(chunks):,} chunks embedded with {MODEL_EMBED} ({DIMS} numbers each)")
print(f"{len(cases)} document questions; right = a chunk from the answer's document in the top 5\n")
print(f"  {'query vectors from':<44}{'right':>7}{'top score':>11}")
right, top = search(vectors, embed(MODEL_EMBED, questions))
print(f"  {MODEL_EMBED + ' (the index model)':<44}{right:>4}/{len(cases)}{top:>11.2f}")
for model, dimensions in fits:
    label = model + (f", dimensions={dimensions}" if dimensions else "")
    right, top = search(vectors, embed(model, questions, dimensions))
    print(f"  {label:<44}{right:>4}/{len(cases)}{top:>11.2f}")
sizes = [sum(ch["source"] == c["source"] for ch in chunks) for c in cases]
chance = sum(1 - math.comb(len(chunks) - n, 5) / math.comb(len(chunks), 5) for n in sizes)
print(f"  {'five chunks drawn at random (expected)':<44}{chance:>7.1f}")
try:
    search(vectors, embed(MODEL_EMBED, questions, 512))
except ValueError as error:
    sizes_differ = re.search(r"size \d+ is different from \d+", str(error)).group(0)
    print(f"  {MODEL_EMBED + ', dimensions=512':<44}")
    print(f"    ValueError: {sizes_differ}")

# A migration left half done: every other document re-embedded with the first model that fits.
new_model, new_dims = fits[0] if fits else (MODEL_EMBED, None)
sources = sorted({c["source"] for c in chunks})
moved = set(sources[::2])
mixed = vectors.copy()
rows = [i for i, c in enumerate(chunks) if c["source"] in moved]
mixed[rows] = embed(new_model, [chunks[i]["text"] for i in rows], new_dims)
print(f"\nhalf migrated: every other document, {len(moved)} of {len(sources)} "
      f"({len(rows):,} chunks),")
print(f"re-embedded with {new_model}")
print(f"  {'query vectors from':<26}{'answer in a':>13}{'answer in a':>13}{'top score':>11}")
print(f"  {'':<26}{'moved doc':>13}{'kept doc':>13}")
in_moved = [c["source"] in moved for c in cases]
for model, dimensions in ((MODEL_EMBED, None), (new_model, new_dims)):
    queries = embed(model, questions, dimensions)
    scores = queries @ mixed.T
    top = np.argsort(-scores, axis=1)[:, :5]
    hit = [any(chunks[i]["source"] == c["source"] for i in row) for row, c in zip(top, cases)]
    moved_hits = sum(h for h, m in zip(hit, in_moved) if m)
    kept_hits = sum(h for h, m in zip(hit, in_moved) if not m)
    moved_cell = f"{moved_hits}/{sum(in_moved)}"
    kept_cell = f"{kept_hits}/{len(cases) - sum(in_moved)}"
    top_score = statistics.median(scores.max(axis=1))
    print(f"  {model:<26}{moved_cell:>13}{kept_cell:>13}{top_score:>11.2f}")
