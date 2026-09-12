# timeout: 1200
# Clarity v0.6, and the question Chapter 13 could not answer.

import sys

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever      # noqa: E402
from meridian_index import load_index            # noqa: E402

chunks, vectors = load_index()
retriever = Retriever(chunks, vectors)

QUESTIONS = [
    "What was total revenue in 2024 Q3?",
    "What is the price adjustment cap in agreement MSC-2022-100?",
    "How long do we have to settle an invoice?",
]

for question in QUESTIONS:
    facets = retriever.facets(question)
    passages = retriever.search(question, k=4)
    named = {k: v for k, v in facets.model_dump().items() if v}
    print(f"Q: {question}")
    print(f"   inferred filter: {named or 'none'}")
    for p in passages:
        print(f"     {p.score:.3f}  {p.source:26} {p.heading[:34]}")
    print()

print("The first question is the one Chapter 13 got wrong, where the right chunk ranked")
print("ninth behind eight other quarters. Nothing about the corpus or the embedding")
print("changed. The question now carries a filter it did not have to be given.")
