# Meaning as a list of numbers, and what "close" turns out to mean.

import numpy as np
from openai import OpenAI

from clarity.config import MODEL_EMBED

client = OpenAI()

TERMS = [
    "cleaning cloths", "microfibre wipes", "janitorial supplies",
    "safety goggles", "protective gloves", "hard hat",
    "invoice", "payment terms", "purchase order",
    "Southwest region", "Midwest region",
]

response = client.embeddings.create(model=MODEL_EMBED, input=TERMS)
vectors = np.array([item.embedding for item in response.data])

print(f"{len(TERMS)} terms, each now a list of {vectors.shape[1]} numbers")
print(f"First five numbers of {TERMS[0]!r}:")
print(f"   {np.round(vectors[0][:5], 4).tolist()}")

# Cosine similarity: 1.0 is identical direction, 0.0 is unrelated.
normalised = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
similarity = normalised @ normalised.T

print("\nClosest neighbour of each term")
for i, term in enumerate(TERMS):
    order = np.argsort(-similarity[i])
    nearest = order[1]          # order[0] is the term itself
    print(f"  {term:22} -> {TERMS[nearest]:22} {similarity[i][nearest]:.3f}")

print("\nA pair the numbers do NOT separate")
a, b = TERMS.index("Southwest region"), TERMS.index("Midwest region")
print(f"  {TERMS[a]!r} vs {TERMS[b]!r}: {similarity[a][b]:.3f}")
print("  Two different places, nearly the same vector. Embeddings capture the kind")
print("  of thing something is, not which particular one. Chapter 11 is about when")
print("  that helps and Chapter 14 about when it quietly ruins your retrieval.")

# Save the 2D projection so the book can plot exactly these vectors.
centred = vectors - vectors.mean(axis=0)
_, _, vt = np.linalg.svd(centred, full_matrices=False)
coords = centred @ vt[:2].T
np.save("code/05/_embedding_2d.npy", coords)
with open("code/05/_embedding_terms.txt", "w") as f:
    f.write("\n".join(TERMS))
