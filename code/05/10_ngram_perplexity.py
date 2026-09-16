# A language model is a probability for what comes next. The oldest kind counts: how often
# did this word follow the previous one? Here are three of them, trained on Meridian's own
# documents and scored by the number every language model is trained to lower.

import json
import math
import re
from collections import Counter
from pathlib import Path

DOCS = Path("data/meridian/documents")
HELD_OUT = {"qbr-2025-Q3.md", "qbr-2025-Q4.md"}


def words(text: str) -> list[str]:
    return ["<s>"] + re.findall(r"[a-z]+|\d+|[^\sa-z\d]", text.lower()) + ["</s>"]


train, test = [], []
for path in sorted((DOCS / "quarterly-reviews").glob("*.md")):
    (test if path.name in HELD_OUT else train).extend(words(path.read_text()))
for path in sorted((DOCS / "contracts").glob("*.md")):
    train.extend(words(path.read_text()))
for line in (DOCS / "tickets" / "tickets.jsonl").read_text().splitlines():
    train.extend(words(json.loads(line)["body"]))

vocab = set(train) | {"<unk>"}
test = [w if w in vocab else "<unk>" for w in test]
V = len(vocab)
uni = Counter(train)
bi = Counter(zip(train, train[1:]))
tri = Counter(zip(train, train[1:], train[2:]))
K = 0.01                                          # add-k smoothing: never a zero


def p_uniform(context, w):
    return 1 / V


def p_unigram(context, w):
    return (uni[w] + K) / (len(train) + K * V)


def p_bigram(context, w):
    prev = context[-1]
    return (bi[(prev, w)] + K) / (uni[prev] + K * V)


def p_trigram(context, w):
    a, b = context[-2], context[-1]
    return (tri[(a, b, w)] + K) / (bi[(a, b)] + K * V)


def perplexity(p) -> tuple[float, float]:
    """exp of the average negative log-probability per word: the cross-entropy, exponentiated."""
    total = 0.0
    for i in range(2, len(test)):
        total -= math.log(p(test[:i][-2:], test[i]))
    n = len(test) - 2
    return total / n, math.exp(total / n)


print(f"trained on {len(train):,} words; vocabulary {V:,}; scored on {len(test) - 2:,} "
      "held-out words\n")
print(f"  {'model':26} {'cross-entropy':>13} {'perplexity':>11}")
results = {}
for name, fn in [("uniform (no information)", p_uniform), ("unigram (word frequency)", p_unigram),
                 ("bigram (one word back)", p_bigram), ("trigram (two words back)", p_trigram)]:
    ce, ppl = perplexity(fn)
    results[name] = ppl
    print(f"  {name:26} {ce:>9.2f} nats {ppl:>11,.1f}")

best = min(results, key=results.get)
print(f"\nlowest perplexity: {best}")
print(f"the uniform model is as unsure as a choice among {results['uniform (no information)']:,.0f} "
      f"words; the best is as unsure as a choice among {results[best]:,.0f}")

# And what a counting model does with its probabilities: generate.
prev, out = "<s>", []
followers = {}
for (a, b), n in bi.items():
    followers.setdefault(a, []).append((n, b))
for _ in range(18):
    prev = max(followers.get(prev, [(0, "</s>")]))[1]        # greedy: the likeliest next word
    if prev == "</s>":
        break
    out.append(prev)
print("\ngreedy bigram text:", " ".join(out))
