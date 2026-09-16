# timeout: 600
# Depends on the Meridian index (meridian_index.py) and the golden set.
# The narrow cone, straightened: subtract the corpus's mean vector (and, optionally, its top few
# principal directions) before comparing. What it does to the spread of similarities, to hubs,
# and to whether the verified chunk is found.

import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                          # noqa: E402
from clarity.evals.runner import load, plain                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

K = 5
chunks, vectors = load_index()
texts = [plain(" ".join(c["text"].split())) for c in chunks]
cases = []
for case in load():
    probe = plain(" ".join((case.get("span") or "").split())[:60])
    gold = {i for i, t in enumerate(texts) if probe and probe in t}
    if gold:
        cases.append((case["question"], gold))
queries = np.array([d.embedding for d in OpenAI().embeddings.create(
    model=MODEL_EMBED, input=[q for q, _ in cases]).data])

mean = vectors.mean(axis=0)
_, _, directions = np.linalg.svd(vectors - mean, full_matrices=False)


def transform(x: np.ndarray, centre: bool, drop: int) -> np.ndarray:
    x = x - mean if centre else x.copy()
    if drop:
        top = directions[:drop]
        x = x - (x @ top.T) @ top
    return x / np.linalg.norm(x, axis=1, keepdims=True)


rng = np.random.default_rng(11)
pairs = rng.integers(0, len(chunks), (5_000, 2))
pairs = pairs[pairs[:, 0] != pairs[:, 1]]
print(f"{len(chunks):,} chunks, {len(cases)} golden questions with a verified chunk; 'biggest hub' is the")
print("most times one chunk appears in the other chunks' ten nearest neighbours\n")
print(f"  {'':28}{'random pair':>12}{'biggest':>8}{'':>9}{'':>9}{'at 5 vs raw':>12}")
print(f"  {'vectors':28}{'mean sim':>12}{'hub':>8}{'recall@1':>9}{'recall@5':>9}{'won, lost':>12}")
baseline = None
for label, centre, drop in (("as returned", False, 0), ("mean subtracted", True, 0),
                            ("mean and top 1 direction", True, 1),
                            ("mean and top 3 directions", True, 3),
                            ("mean and top 10 directions", True, 10)):
    v, qv = transform(vectors, centre, drop), transform(queries, centre, drop)
    cone = float(np.mean(np.sum(v[pairs[:, 0]] * v[pairs[:, 1]], axis=1)))
    sims = v @ v.T
    np.fill_diagonal(sims, -np.inf)
    neighbours = np.argpartition(-sims, 10, axis=1)[:, :10]
    hub = np.bincount(neighbours.ravel(), minlength=len(chunks)).max()
    ranked = np.argsort(-(qv @ v.T), axis=1)
    found = lambda k: np.array([bool(set(r[:k]) & g) for r, (_, g) in zip(ranked, cases)])  # noqa: E731
    at5 = found(K)
    baseline = at5 if baseline is None else baseline
    change = f"+{np.sum(at5 & ~baseline)}, -{np.sum(~at5 & baseline)}"
    print(f"  {label:28}{cone:>12.3f}{hub:>8}{found(1).mean():>9.0%}{at5.mean():>9.0%}{change:>12}")
