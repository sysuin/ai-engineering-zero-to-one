# timeout: 1200
# Retrieval "missed" is not one diagnosis. Where the right passage actually ranked says
# which fix to try: a bigger k, a reranker, or a different query or index.

import sys

sys.path.insert(0, "code")
from clarity.evals.runner import load, plain                       # noqa: E402
from clarity.v0_5.rag import Clarity                               # noqa: E402
from clarity.v0_6.retrieve import Retriever                        # noqa: E402
from meridian_index import load_index                              # noqa: E402

DEPTH, K = 50, 6
chunks, vectors = load_index()
dense_only = Clarity(chunks, vectors)
hybrid = Retriever(chunks, vectors)
cases = [c for c in load()
         if c["kind"] == "document" and c.get("span") and c.get("tier") == "tail"]


def rank(texts: list[str], span: str) -> int | None:
    probe = plain(" ".join(span.split())[:60])
    for position, text in enumerate(texts, 1):
        if probe in plain(" ".join(text.split())):
            return position
    return None


def verdict(position: int | None) -> str:
    if position is None:
        return f"not in top {DEPTH}"
    if position <= K:
        return "in the top k"
    return "just below k" if position <= 20 else "deep"


print(f"{len(cases)} questions whose answer is in one passage; where that passage ranked\n")
print(f"  {'question':<44} {'dense (v0.5)':>13} {'hybrid (v0.6)':>14}")
tally = {"dense": {}, "hybrid": {}}
rescued_questions = []
for case in cases:
    a = rank([p.text for p in dense_only.retrieve(case["question"], k=DEPTH)], case["span"])
    b = rank([p.text for p in hybrid.search(case["question"], k=DEPTH)], case["span"])
    if verdict(a) != "in the top k" and verdict(b) == "in the top k":
        rescued_questions.append(case["question"])
    for name, position in (("dense", a), ("hybrid", b)):
        tally[name][verdict(position)] = tally[name].get(verdict(position), 0) + 1
    show = lambda p: "—" if p is None else str(p)                   # noqa: E731
    print(f"  {case['question'][:42]:<44} {show(a):>13} {show(b):>14}")

print(f"\n  {'':<22} {'dense':>6} {'hybrid':>7}   what to try")
advice = {"in the top k": "retrieval is not the problem",
          "just below k": "a larger k, or a reranker over the top 20",
          "deep": "the query or the chunking, not k",
          f"not in top {DEPTH}": "the index: is the text even chunked findably?"}
for label, fix in advice.items():
    print(f"  {label:<22} {tally['dense'].get(label, 0):>6} "
          f"{tally['hybrid'].get(label, 0):>7}   {fix}")

missed = len(cases) - tally["dense"].get("in the top k", 0)
absent = tally["dense"].get(f"not in top {DEPTH}", 0)
print(f"\nDense retrieval missed {missed} of these at k={K}; "
      f"{'none' if absent == 0 else absent} of them "
      f"{'was' if absent in (0, 1) else 'were'} missing from the top {DEPTH}.")
print(f"Hybrid retrieval brought {len(rescued_questions)} into the top {K}:")
for question in rescued_questions:
    print(f"  {question}")
print("'Retrieval failed' was several different failures, and each has a different fix.")
