# timeout: 900
# Two things the chapter argued without measuring. How much a reranker could ever win on the
# unfiltered hybrid — the gap between what is in its pool and what reaches the top five — and
# whether searching with several phrasings of a question finds what one phrasing missed.

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _retrieval import QUESTIONS, TRUTH, bm25_order, client, embed, recall_at, rrf, vectors
from clarity.config import MODEL_FAST

TAGS = ("plain", "near-duplicate", "identifier", "paraphrase", "ticket", "obscure")


def dense_order(q: np.ndarray, depth: int = 50) -> list[int]:
    return [int(i) for i in np.argsort(-(vectors @ q))[:depth]]


def hybrid(q: np.ndarray, text: str) -> list[int]:
    return rrf(dense_order(q), bm25_order(text)[:50], k=1)


Q = embed([q for q, _, _ in QUESTIONS])
single = [hybrid(q, text) for q, (text, _, _) in zip(Q, QUESTIONS)]

DEPTHS = (1, 3, 5, 10, 25, 50)
print("How much was left for a reranker: recall at each depth\n")
print("  " + "".join(f"{'@' + str(k):>7}" for k in DEPTHS))
print("  " + "".join(f"{recall_at(single, k):>7.0%}" for k in DEPTHS))
in_pool = recall_at(single, 25) - recall_at(single, 5)
gap = round(in_pool * len(QUESTIONS))
never = sum(not (set(o[:25]) & t) for o, t in zip(single, TRUTH))
print(f"\n  a reranker over the top 25 could win at most {gap};")
print(f"  {never} have no answering chunk in the 25 to promote")


# Several phrasings, from one call per question.
ASK = ("Rewrite this question three different ways that a person "
       "searching company documents might phrase it, each worded "
       "differently from the original. Keep every name, number and "
       "identifier. Reply with a JSON list of three strings and "
       "nothing else.")


def phrasings(question: str) -> list[str]:
    text = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200,
        messages=[{"role": "user", "content": f"{ASK}\n\n{question}"}],
    ).choices[0].message.content or "[]"
    try:
        return [p for p in json.loads(text) if isinstance(p, str)][:3]
    except json.JSONDecodeError:
        return []


def same(a: str, b: str) -> bool:
    return a.strip().lower() == b.strip().lower()


def one_run():
    with ThreadPoolExecutor(max_workers=10) as pool:
        returned = list(pool.map(lambda q: phrasings(q[0]), QUESTIONS))
    # A "rewrite" that repeats the question adds nothing but weight: count, then drop.
    repeats = sum(any(same(p, q) for p in ps) for (q, _, _), ps in zip(QUESTIONS, returned))
    rewrites = [[p for p in ps if not same(p, q)] for (q, _, _), ps in zip(QUESTIONS, returned)]
    P = embed([p for ps in rewrites for p in ps])
    vectors_for, start = [], 0
    for ps in rewrites:
        vectors_for.append(P[start:start + len(ps)])
        start += len(ps)
    multi_dense = [rrf(dense_order(q), *(dense_order(v) for v in vs), k=1)
                   for q, vs in zip(Q, vectors_for)]
    multi_hybrid = [rrf(*(hybrid(v, p) for v, p in zip([q, *vs], [text, *ps])), k=1)
                    for q, vs, (text, _, _), ps in zip(Q, vectors_for, QUESTIONS, rewrites)]
    only_rewrites = [rrf(*(hybrid(v, p) for v, p in zip(vs, ps)), k=1) if ps else []
                     for vs, ps in zip(vectors_for, rewrites)]
    return repeats, rewrites, multi_dense, multi_hybrid, only_rewrites, returned


runs = [one_run() for _ in range(3)]
repeats, rewrites, multi_dense, multi_hybrid, only_rewrites, returned = runs[0]
dense_single = [dense_order(q) for q in Q]


def row(name: str, orders: list[list[int]]) -> None:
    cells = ""
    for tag in TAGS:
        idx = [i for i, (_, _, t) in enumerate(QUESTIONS) if t == tag]
        cells += f"{sum(bool(set(orders[i][:5]) & TRUTH[i]) for i in idx) / len(idx):>7.0%}"
    print(f"  {name:<30}{recall_at(orders):>5.0%}{cells}")


print(f"\nSeveral phrasings: {sum(map(len, rewrites))} distinct rewrites for "
      f"{len(QUESTIONS)} questions")
print(f"  (the model repeated the question as a 'rewrite' for {repeats} of them)\n")
print(f"  {'recall@5':<30}{'all':>5}" + "".join(f"{t[:6]:>7}" for t in TAGS))
row("dense, the question", dense_single)
row("dense, question + rewrites", multi_dense)
row("hybrid, the question", single)
row("hybrid, question + rewrites", multi_hybrid)
row("hybrid, the rewrites only", only_rewrites)


def hit(orders: list[list[int]], i: int) -> bool:
    return bool(set(orders[i][:5]) & TRUTH[i])


missed = [i for i in range(len(QUESTIONS)) if not hit(single, i)]
print(f"\n  the {len(missed)} questions the hybrid missed, and which variant answered them")
print(f"    {'':<52}{'+ rewrites':>11}{'only':>6}")
for i in missed:
    marks = ["yes" if hit(o, i) else "-" for o in (multi_hybrid, only_rewrites)]
    print(f"    {QUESTIONS[i][2][:5]:<6}{QUESTIONS[i][0][:46]:<46}{marks[0]:>11}{marks[1]:>6}")
for name, orders in (("question + rewrites", multi_hybrid), ("rewrites only", only_rewrites)):
    lost = sum(hit(single, i) and not hit(orders, i) for i in range(len(QUESTIONS)))
    print(f"  questions the hybrid answered and {name} did not: {lost}")
print("\n  the same, generating the rewrites again (recall@5 on each of three runs)")
for name, k in (("hybrid, question + rewrites", 3), ("hybrid, the rewrites only", 4)):
    print(f"    {name:<30}" + "".join(f"{recall_at(run[k]):>6.0%}" for run in runs))
print(f"    {'repeated the question':<30}" + "".join(f"{run[0]:>6}" for run in runs))
example = missed[0]
print(f"\n  what the model returned for {QUESTIONS[example][0]!r}:")
for p in returned[example]:
    print(f"    {p}" + ("   (the question again: dropped)" if same(p, QUESTIONS[example][0]) else ""))
