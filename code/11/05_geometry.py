# The geometry of embeddings: an identity worth knowing, a narrow cone, and what happens to
# distances as dimensions grow.

import numpy as np
from openai import OpenAI

from _corpus import load
from clarity.config import MODEL_EMBED

client = OpenAI()
corpus = load()
data = client.embeddings.create(model=MODEL_EMBED, input=[c["text"] for c in corpus]).data
V = np.array([d.embedding for d in data])
print("vector lengths as returned:", f"{np.linalg.norm(V, axis=1).min():.4f} to "
      f"{np.linalg.norm(V, axis=1).max():.4f}")
V = V / np.linalg.norm(V, axis=1, keepdims=True)

# 1. For unit vectors, squared distance and cosine carry the same information.
a, b = V[0], V[10]
print(f"\n|a - b|^2 = {np.sum((a - b) ** 2):.6f}    2 - 2 cos(a, b) = {2 - 2 * a @ b:.6f}")

# 2. The cone: even unrelated clauses have clearly positive similarity.
S = V @ V.T
same, different = [], []
for i in range(len(corpus)):
    for j in range(i + 1, len(corpus)):
        (same if corpus[i]["clause"] == corpus[j]["clause"] else different).append(S[i, j])
print(f"\nsimilarity between clauses on the SAME topic (different contracts):  "
      f"mean {np.mean(same):.3f}, lowest {np.min(same):.3f}")
print(f"similarity between clauses on DIFFERENT topics:                      "
      f"mean {np.mean(different):.3f}, lowest {np.min(different):.3f}")
print(f"share of different-topic pairs above zero: {np.mean(np.array(different) > 0):.0%}")

# 3. Distance concentration, with random points: in high dimensions, the nearest and farthest
#    neighbours of a point end up at nearly the same distance.
rng = np.random.default_rng(1)
print("\nrandom points: (farthest - nearest) / nearest distance from one point to 1,000 others")
for d in (2, 16, 128, 1536):
    X = rng.normal(size=(1001, d))
    dist = np.linalg.norm(X[1:] - X[0], axis=1)
    print(f"  {d:>5} dimensions: {(dist.max() - dist.min()) / dist.min():.2f}")
