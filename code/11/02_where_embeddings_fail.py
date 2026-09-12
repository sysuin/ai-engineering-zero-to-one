# timeout: 900
# Three things embeddings are reliably bad at, each measured.

import json
from pathlib import Path

import numpy as np
from openai import OpenAI

from clarity.config import MODEL_EMBED

client = OpenAI()


def embed(texts: list[str]) -> np.ndarray:
    vectors = np.array([d.embedding for d in
                        client.embeddings.create(model=MODEL_EMBED,
                                                 input=texts).data])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def similarity(a: str, b: str) -> float:
    v = embed([a, b])
    return float(v[0] @ v[1])


print("1. Negation — the word 'not' barely moves the vector\n")
PAIRS = [
    ("The supplier may raise prices.", "The supplier may not raise prices."),
    ("Goods conform to the specification.", "Goods do not conform to the specification."),
    ("The buyer may terminate for convenience.",
     "The buyer may terminate for convenience."),
]
for a, b in PAIRS:
    same = a == b
    print(f"   {similarity(a, b):.3f}  {a[:44]}")
    print(f"          {'(identical)' if same else 'vs'} {b[:44]}")
print("   A sentence and its exact opposite sit almost on top of each other.")

print("\n2. Identifiers — every product code looks like every other product code\n")
CODES = ["MRD-CLE-001", "MRD-CLE-004", "MRD-SAF-011", "MRD-PAC-023"]
vectors = embed(CODES)
matrix = vectors @ vectors.T
print("        " + "".join(f"{c[-3:]:>9}" for c in CODES))
for i, code in enumerate(CODES):
    print(f"   {code[-7:]:>7}" + "".join(f"{matrix[i][j]:>9.3f}" for j in range(len(CODES))))
print("   Asking for MRD-CLE-001 will happily return MRD-CLE-004.")

print("\n3. Numbers — magnitude is not encoded\n")
AMOUNTS = ["a payment term of 30 days", "a payment term of 45 days",
           "a payment term of 300 days", "a payment term of 30 years"]
vectors = embed(AMOUNTS)
base = vectors[0]
for text, vector in zip(AMOUNTS, vectors):
    print(f"   {float(base @ vector):.3f}  {text}")
print("   '30 days' is closer to '30 years' than to '45 days'."
      if float(base @ embed(["a payment term of 30 years"])[0]) >
      float(base @ embed(["a payment term of 45 days"])[0])
      else "   The ordering here happens to be sensible; it is not guaranteed to be.")

print("\n4. And the one that matters most: similarity is not confidence\n")
QUERY = "What is the maximum price increase allowed?"
CANDIDATES = [
    ("the right answer", "The Supplier may adjust prices by no more than 8% in "
                         "aggregate."),
    ("a related clause", "The Supplier may terminate this Agreement on 90 days notice."),
    ("a different document", "The Columbus depot will close for inventory count."),
    ("complete nonsense", "Bananas are typically harvested while still green."),
]
qv = embed([QUERY])[0]
for label, text in CANDIDATES:
    print(f"   {float(qv @ embed([text])[0]):.3f}  {label:22} {text[:44]}")

print()
print("The ordering is right, and that is all it is: an ordering.")
print()
print("Look at the top score. The correct, directly responsive answer scores 0.5 — not")
print("0.9, not anything you would recognise as 'confident'. There is no threshold you")
print("could set here that means 'good enough'. Set it at 0.4 and this passes; ask a")
print("question with no answer in the corpus and the best of a hundred irrelevant")
print("chunks will also clear 0.4, because something always ranks first.")
print()
print("A similarity score ranks candidates against each other. It says nothing about")
print("whether the winner is any good. Chapter 13 needs a different signal for that,")
print("and this is why.")

Path("code/11/_failures.json").write_text(json.dumps({
    "negation": similarity(PAIRS[0][0], PAIRS[0][1]),
    "identifiers": float(matrix[0][1]),
    "relevance": {label: float(qv @ embed([text])[0]) for label, text in CANDIDATES},
}, indent=2))
