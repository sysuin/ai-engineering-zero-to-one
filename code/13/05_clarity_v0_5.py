# timeout: 900
# Clarity v0.5 answering questions across the whole Meridian corpus.

import sys

sys.path.insert(0, "code")
from clarity.v0_5.rag import Clarity      # noqa: E402
from meridian_index import load_index     # noqa: E402

chunks, vectors = load_index()
clarity = Clarity(chunks, vectors)
print(f"{len(chunks):,} chunks indexed\n")

QUESTIONS = [
    ("Which account did not renew, and what does the company say caused it?", None),
    ("What was total revenue in 2024 Q3?", None),
    ("What was total revenue in 2024 Q3?",
     {"kind": "quarterly-review", "year": 2024, "quarter": 3}),
    ("How many people work at the Columbus depot?", None),
]

for question, where in QUESTIONS:
    result = clarity.answer(question, where=where)
    print(f"Q: {question}")
    if where:
        print(f"   filter: {where}")
    print(f"   {'answered' if result.grounded else 'ABSTAINED'}"
          f"   sources: {', '.join(result.sources[:4])}")
    print(f"   {result.text[:300]}")
    print()

print("The second and third questions are identical. The only difference is a metadata")
print("filter, and it is the difference between a wrong answer and a right one.")
print()
print("Without it, retrieval returns the Summary section of eight other quarters —")
print("they are all worded the same way — and Clarity correctly reports that it cannot")
print("answer. The abstention gate turned a silent wrong answer into a visible gap.")
print()
print("Chapter 14 is about not needing the filter to be supplied by hand.")
