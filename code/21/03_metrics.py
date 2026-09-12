# timeout: 900
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# Reads code/21/_scorecard.json, written by 01. Re-run that first if the golden
# set or the scorer changes — this file grades the answers 01 recorded.
# Four metrics, one set of answers, four different scores.

import json
import re
import sys

import numpy as np

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                          # noqa: E402
from clarity.evals.runner import abstained, load, normalise, plain  # noqa: E402
from openai import OpenAI                                       # noqa: E402

client = OpenAI()
cases = {c["id"]: c for c in load()}
rows = [r for r in json.load(open("code/21/_scorecard.json"))
        if r["kind"] != "unanswerable"]


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9.]+", plain(text)))


def exact(case: dict, answer: str) -> bool:
    """The whole reply must be the answer. Nobody's system passes this."""
    return normalise(answer) == normalise(case["answer"])


def contains(case: dict, answer: str) -> bool:
    """The answer appears somewhere in the reply. This is what §21.1 used."""
    wanted = [case["answer"]] + list(case.get("accept") or [])
    return any(normalise(w) in normalise(answer) for w in wanted if w)


def token_f1(case: dict, answer: str) -> float:
    """The standard extractive-QA metric: overlap between two bags of words."""
    gold, got = tokens(case["answer"]), tokens(answer)
    if not gold or not got:
        return 0.0
    shared = len(gold & got)
    if not shared:
        return 0.0
    precision, recall = shared / len(got), shared / len(gold)
    return 2 * precision * recall / (precision + recall)


def embed(texts: list[str]) -> np.ndarray:
    out = client.embeddings.create(model=MODEL_EMBED, input=texts)
    vectors = np.array([d.embedding for d in out.data])
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


gold = embed([cases[r["id"]]["answer"] for r in rows])
got = embed([r["answer"][:1500] for r in rows])
similarity = (gold * got).sum(axis=1)

scores = {"exact match": [], "contains the answer": [], "token F1": [],
          "embedding similarity": []}
for row, sim in zip(rows, similarity):
    case = cases[row["id"]]
    scores["exact match"].append(float(exact(case, row["answer"])))
    scores["contains the answer"].append(float(contains(case, row["answer"])))
    scores["token F1"].append(token_f1(case, row["answer"]))
    scores["embedding similarity"].append(float(sim))

print(f"{len(rows)} answerable cases, one set of replies, four ways of scoring.\n")
print(f"  {'metric':<24}{'score':>8}   what it counts as right")
notes = {"exact match": "the reply is exactly the answer",
         "contains the answer": "the answer is somewhere in the reply",
         "token F1": "word overlap with the answer",
         "embedding similarity": "the reply means something similar"}
for name, values in scores.items():
    print(f"  {name:<24}{sum(values) / len(values):>7.0%}   {notes[name]}")

json.dump({name: values for name, values in scores.items()} |
          {"ids": [r["id"] for r in rows]},
          open("code/21/_metrics.json", "w"), indent=1)

# Similarity is a number, not a verdict. To use it as one you need a threshold, so
# find the best threshold this data allows and see how good the verdict can get.
truth = [bool(v) for v in scores["contains the answer"]]
sims = scores["embedding similarity"]
best = max(
    ((sum((s >= t) == y for s, y in zip(sims, truth)) / len(sims), t)
     for t in [i / 100 for i in range(20, 96)]))
best_accuracy, best_threshold = best
right = [s for s, y in zip(sims, truth) if y]
wrong = [s for s, y in zip(sims, truth) if not y]
overlap = sum(1 for s in right if s <= max(wrong)) if wrong else 0

print()
print(f"Exact match scores {sum(scores['exact match']) / len(rows):.0%}, and the "
      f"system is not that bad —")
print("it answers in sentences, and a sentence is never equal to a number. Exact")
print("match is the right metric only when the output has been constrained to be")
print("exactly the answer, which Chapter 8 shows how to do and this system does not.")
print()
print(f"Token F1 gives {sum(scores['token F1']) / len(rows):.0%}, which looks like a "
      f"score and is not one. It rewards a")
print("long reply for containing common words and punishes a correct short one for")
print("containing few.")
print()
print("Embedding similarity is the interesting failure. Look at what it does to the")
print(f"{len(right)} answers the string rule calls correct:\n")
print(f"  lowest similarity among correct answers   {min(right):.2f}")
print(f"  highest                                   {max(right):.2f}")
print(f"  mean                                      {sum(right) / len(right):.2f}")
if wrong:
    print(f"  mean among wrong answers                  "
          f"{sum(wrong) / len(wrong):.2f}   (only {len(wrong)} of them)")
print()
print(f"Correct answers span {min(right):.2f} to {max(right):.2f}, and the number is "
      f"not comparable across")
print("questions: a short exact reply to a long question scores low because it is")
print("short, not because it is wrong.")
print()
print(f"This run produced only {len(wrong)} wrong answers, which is far too few to fit "
      f"a threshold")
print("to. That is worth saying rather than quietly fitting one anyway — with a")
print("handful of negatives, any cut-off you compute describes a handful of")
print("sentences.")
print()
print("Two correct answers from the bottom of the range:\n")
correct_low = [i for i in sorted(range(len(rows)), key=lambda i: sims[i])
               if truth[i] and rows[i]["kind"] == "document"]
for i in correct_low[:2]:
    flat = " ".join(rows[i]["answer"].split())
    print(f"  scored {sims[i]:.2f}")
    print(f"    Q: {rows[i]['question'][:64]}")
    print(f"    A: {flat[:64]}")
    print(f"    gold: {cases[rows[i]['id']]['answer'][:40]}")
print()
print("Similarity measures topic and register, not truth. Use it to rank near misses")
print("worth reading. Do not use it to decide whether you shipped a regression.")
