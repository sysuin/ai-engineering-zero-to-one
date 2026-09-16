# How a tokeniser is built: byte-pair encoding, from scratch, on Meridian's text.
# Start from single characters and repeatedly glue together the most frequent adjacent pair.

import re
from collections import Counter
from pathlib import Path

text = " ".join(p.read_text() for p in sorted(
    Path("data/meridian/documents/quarterly-reviews").glob("*.md")))
# Each word, with a marker for the space before it, as a tuple of symbols.
# Words and numbers only: the Markdown table rules would otherwise win the first merges.
corpus = Counter(tuple("▁" + w) for w in re.findall(r"[A-Za-z0-9$%,.'’]+(?:-[A-Za-z0-9]+)*", text))
MERGES = 300


def pair_counts(corpus):
    pairs = Counter()
    for word, n in corpus.items():
        for a, b in zip(word, word[1:]):
            pairs[(a, b)] += n
    return pairs


def merge(corpus, pair):
    joined = pair[0] + pair[1]
    out = Counter()
    for word, n in corpus.items():
        new, i = [], 0
        while i < len(word):
            if i + 1 < len(word) and (word[i], word[i + 1]) == pair:
                new.append(joined)
                i += 2
            else:
                new.append(word[i])
                i += 1
        out[tuple(new)] += n
    return out


merges = []
for step in range(MERGES):
    pairs = pair_counts(corpus)
    best, count = pairs.most_common(1)[0]
    corpus = merge(corpus, best)
    merges.append(best)
    if step < 8 or step in (49, 149, 299):
        print(f"merge {step + 1:>3}: {best[0]!r} + {best[1]!r} -> {best[0] + best[1]!r}"
              f"   ({count} times)")


def tokenize(word: str) -> list[str]:
    symbols = list("▁" + word)
    for a, b in merges:                     # apply the merges in the order they were learned
        i, out = 0, []
        while i < len(symbols):
            if i + 1 < len(symbols) and symbols[i] == a and symbols[i + 1] == b:
                out.append(a + b)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        symbols = out
    return symbols


print()
for word in ["revenue", "Midwest", "Meridian", "quarterly", "Halloway", "8,461,842", "zymurgy"]:
    print(f"  {word:>10} -> {tokenize(word)}")
