# BM25 can say why it ranked a clause first: its score is a sum over query words, so each
# word's share can be printed. The questions it got wrong, taken apart word by word.

import math
import re
from collections import Counter

from _corpus import QUERIES, load

corpus = load()
K1, B = 1.5, 0.75


def tokenise(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]{3,}", text.lower())


DOCS = [tokenise(c["text"]) for c in corpus]
DF = Counter(w for d in DOCS for w in set(d))
AVG = sum(len(d) for d in DOCS) / len(DOCS)


def explain(query: str, i: int) -> dict[str, float]:
    """Each query word's contribution to document i's score."""
    counts, parts = Counter(DOCS[i]), {}
    for term in dict.fromkeys(tokenise(query)):
        if term in counts:
            df = DF[term]
            idf = math.log(1 + (len(DOCS) - df + 0.5) / (df + 0.5))
            tf = counts[term]
            norm = K1 * (1 - B + B * len(DOCS[i]) / AVG)
            parts[term] = idf * tf * (K1 + 1) / (tf + norm)
    return parts


def show(label: str, query: str, i: int) -> None:
    parts = explain(query, i)
    words = ", ".join(f"{t} {s:.2f} (in {DF[t]})" for t, s in
                      sorted(parts.items(), key=lambda p: -p[1])) or "no shared words"
    print(f"  {label:9}cl {corpus[i]['clause']} {corpus[i]['topic']:<28}"
          f"score {sum(parts.values()):.2f}")
    print(f"  {'':9}{words}")


for question, want in QUERIES:
    scores = [sum(explain(question, i).values()) for i in range(len(corpus))]
    top = max(range(len(corpus)), key=scores.__getitem__)
    if corpus[top]["clause"] == want:
        continue
    right = max((i for i, c in enumerate(corpus) if c["clause"] == want),
                key=scores.__getitem__)
    print(f"{question}\n  words: {', '.join(tokenise(question))}")
    show("returned", question, top)
    show("wanted", question, right)
    print()
print(f"each word: its share of the score (in how many of {len(corpus)} clauses it appears)")
