# timeout: 900
# "The user asked again" as a failure signal, measured. Pairs of consecutive questions from one
# session: the same question reworded (the first answer failed), a different follow-up (it did not),
# and the golden set's own neighbours — the same question about another quarter.

import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI
from pydantic import BaseModel

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED, MODEL_FAST              # noqa: E402
from clarity.evals.runner import load                           # noqa: E402

client = OpenAI()
cases = [c for c in load() if c["kind"] != "unanswerable"]
questions = random.Random(23).sample(sorted({c["question"] for c in cases}), 40)


class Next(BaseModel):
    reworded: str       # the same question in other words, from a user whose answer was no use
    retyped: str        # the same question typed again quickly, as frustrated users do
    follow_up: str      # a different question a satisfied user asks next


def next_questions(question: str) -> Next:
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300, response_format=Next,
        messages=[{"role": "system", "content": "You write realistic chat messages from analysts using "
                   "an internal assistant. Given a question, write three messages: (1) what a user types when the "
                   "answer did not help and they ask the same thing again in other words; (2) what a user "
                   "types when the answer did not help and they quickly retype the question, short and "
                   "impatient; (3) the next question a user asks when the answer was fine: a different "
                   "question on the same subject."},
                  {"role": "user", "content": question}]).choices[0].message.parsed


with ThreadPoolExecutor(max_workers=16) as pool:
    generated = list(pool.map(next_questions, questions))

# The golden set's own neighbours: identical except for the year or quarter.
by_template = {}
for q in sorted({c["question"] for c in cases}):
    by_template.setdefault(re.sub(r"\b20\d\d\b|\bQ[1-4]\b", "#", q), []).append(q)
neighbours = [(g[i], g[i + 1]) for g in by_template.values() for i in range(len(g) - 1)]
neighbours = random.Random(23).sample(neighbours, min(40, len(neighbours)))

KINDS = ("reworded", "retyped", "follow-up", "other quarter")
FAILED = {"reworded", "retyped"}
PAIRS = ([(q, n.reworded, "reworded") for q, n in zip(questions, generated)] +
         [(q, n.retyped, "retyped") for q, n in zip(questions, generated)] +
         [(q, n.follow_up, "follow-up") for q, n in zip(questions, generated)] +
         [(a, b, "other quarter") for a, b in neighbours])

texts = sorted({t for a, b, _ in PAIRS for t in (a, b)})
vectors = np.array([d.embedding for d in client.embeddings.create(model=MODEL_EMBED, input=texts).data])
vector = dict(zip(texts, vectors))


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


NUMBER_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve".split())}


def entities(text: str) -> set[str]:
    """What a retry keeps and a new question changes: ids, years,
    quarters, numbers, and names (capitalised, not first word)."""
    text = re.sub(r"\b[a-z]+\b",
                  lambda m: NUMBER_WORDS.get(m[0], m[0]), text)
    ids = re.findall(r"\b[A-Za-z]*-?\d+(?:-\d+)*\b", text)
    names = re.findall(
        r"(?<=[a-z,] )[A-Z][a-z]+(?: [A-Z][a-z]+)*", text)
    return {x.lower() for x in ids + names}


def jaccard(a: str, b: str) -> float:
    return len(words(a) & words(b)) / len(words(a) | words(b))


def cosine(a: str, b: str) -> float:
    return float(vector[a] @ vector[b])


DETECTORS = {
    "shared words (Jaccard)": jaccard,
    "embedding similarity": cosine,
    "similarity, same entities":
        lambda a, b: cosine(a, b) * (entities(a) == entities(b)),
}

print(f"{len(PAIRS)} consecutive pairs from golden-set questions: {len(questions)} each "
      "reworded,\nretyped and followed up (all written by a model), and "
      f"{len(neighbours)} of the same\nquestion about another quarter\n")
WIDTHS = (10, 9, 11, 15)
header = "".join(f"{k:>{w}}" for k, w in zip(KINDS, WIDTHS))


def auc(score) -> float:
    """The chance a randomly chosen failed-answer pair scores above a randomly chosen other pair."""
    pos = [score(a, b) for a, b, k in PAIRS if k in FAILED]
    neg = [score(a, b) for a, b, k in PAIRS if k not in FAILED]
    return np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg])


print(f"  {'mean score':26}{header}{'AUC':>6}")
AUCS = {name: auc(score) for name, score in DETECTORS.items()}
for name, score in DETECTORS.items():
    means = [np.mean([score(a, b) for a, b, k in PAIRS if k == kind]) for kind in KINDS]
    print(f"  {name:26}" + "".join(f"{m:>{w}.2f}" for m, w in zip(means, WIDTHS))
          + f"{AUCS[name]:>6.2f}")

# Each threshold is chosen on half of the source questions, to separate failed answers from the rest
# as evenly as it can (the mean of the two hit rates), and scored on the other half; then the halves
# swap. No detector is graded on the pairs it was tuned on.
half = {q: i % 2 for i, q in enumerate(questions)} | {a: i % 2 for i, (a, _) in enumerate(neighbours)}


def balanced(pairs, t):
    hit = np.mean([s >= t for s, failed in pairs if failed])
    quiet = np.mean([s < t for s, failed in pairs if not failed])
    return (hit + quiet) / 2


print(f"\n  {'flagged as failed':26}{header}")
for name, score in DETECTORS.items():
    flagged = dict.fromkeys(KINDS, 0)
    for fold in (0, 1):
        tune = [(score(a, b), k in FAILED) for a, b, k in PAIRS if half[a] == fold]
        best = max(sorted({s for s, _ in tune}), key=lambda t: balanced(tune, t))
        for a, b, k in PAIRS:
            if half[a] != fold:
                flagged[k] += score(a, b) >= best
    total = {k: sum(1 for *_, kk in PAIRS if kk == k) for k in KINDS}
    print(f"  {name:26}" + "".join(f"{flagged[k] / total[k]:>{w}.0%}" for k, w in zip(KINDS, WIDTHS)))

words_auc, embed_auc, gated_auc = AUCS.values()
print(f"\nAUC 0.50 is a coin toss: shared words {words_auc:.2f}, similarity {embed_auc:.2f}, "
      f"same entities {gated_auc:.2f}")

for q, n in list(zip(questions, generated))[:2]:
    print(f"\n  {q}\n    reworded:  {n.reworded}\n    retyped:   {n.retyped}\n    follow-up: {n.follow_up}")
