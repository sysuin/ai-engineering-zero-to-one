# timeout: 2400
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# A failing case is easier to fix when it is four lines instead of forty.

import json
import sys

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                          # noqa: E402
from clarity.evals.runner import correct, load                 # noqa: E402
from clarity.v0_5.rag import Clarity                           # noqa: E402
from meridian_index import load_index                          # noqa: E402
from openai import OpenAI                                      # noqa: E402

TRIALS = 3            # a failure that happens once is not a failure yet
K = 8
client = OpenAI()
chunks, vectors = load_index()
system = Clarity(chunks, vectors)
ANSWERER = ("Answer the question using only the passages given. Be brief. If the "
            "passages do not contain the answer, say so.")


def answer(question: str, passages: list[str]) -> str:
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=200,
        messages=[{"role": "system", "content": ANSWERER},
                  {"role": "user", "content":
                   "Passages:\n" + "\n\n".join(passages)[:8000] +
                   f"\n\nQuestion: {question}"}])
    return (reply.choices[0].message.content or "").strip()


def fails(case: dict, passages: list[str]) -> bool:
    """Fails means fails repeatedly. §24.1 is why this takes three samples."""
    return sum(not correct(case, answer(case["question"], passages))
               for _ in range(TRIALS)) >= 2


# Find a case that fails with its real retrieved context.
target = None
for case in load():
    if case["kind"] != "document" or not case.get("span"):
        continue
    passages = [p.text for p in system.retrieve(case["question"], k=K)]
    if len(passages) >= 4 and fails(case, passages):
        target = (case, passages)
        break

if target is None:
    print("No case failed repeatedly today. That is a good outcome and a useless")
    print("listing; the shrinking algorithm below is what matters, and §24.5 walks")
    print("through it on a case that did fail.")
    raise SystemExit(0)

case, passages = target
before_chars = sum(len(p) for p in passages)
print(f"Failing case: {case['question'][:62]}")
print(f"  {len(passages)} passages, {before_chars:,} characters, wrong in at least "
      f"2 of {TRIALS} runs\n")

# Delta debugging, in its simplest useful form: drop one passage at a time and keep
# the drop whenever the failure survives it.
kept = list(passages)
removed = 0
for candidate in list(passages):
    if len(kept) == 1:
        break
    trial = [p for p in kept if p is not candidate]
    if fails(case, trial):
        kept = trial
        removed += 1
        print(f"  dropped a passage and it still fails  -> {len(kept)} left")
    else:
        print(f"  dropping a passage fixes it           -> that one matters")

after_chars = sum(len(p) for p in kept)
print(f"\n  {len(passages)} passages -> {len(kept)}   "
      f"{before_chars:,} -> {after_chars:,} characters "
      f"({1 - after_chars / before_chars:.0%} smaller)")
print(f"\n  what is left:\n")
for passage in kept:
    print(f"    {' '.join(passage.split())[:66]}")

json.dump({"id": case["id"], "question": case["question"],
           "before": len(passages), "after": len(kept),
           "before_chars": before_chars, "after_chars": after_chars,
           "kept": [p[:400] for p in kept]},
          open("code/24/_shrink.json", "w"), indent=1)

print()
print("That is delta debugging, and the version above is the crude one — drop each")
print("part, keep the drop if the bug survives. It is O(n) calls and good enough for")
print("a context window; the published algorithm bisects and is worth reading when n")
print("is large.")
print()
print("Two things make it work on a system like this one.")
print()
print("A test that is a *rate*, not a boolean. `fails()` samples three times and asks")
print("for two, because a single run proves nothing — §24.1. Shrink against a flaky")
print("predicate and the algorithm happily removes the thing that was causing the bug.")
print()
print("And shrinking the input, not the system. It is tempting to start disabling")
print("features; that finds which component is involved, which you already knew. The")
print("minimal input tells you what the component cannot handle, which is the thing")
print("you have to fix.")
