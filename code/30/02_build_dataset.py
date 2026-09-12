# timeout: 600
# The dataset, and the four checks that stop you training on nonsense.

import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, "code")
from clarity.evals.runner import load                          # noqa: E402

random.seed(30)
OUT = Path("data/meridian/finetune")
OUT.mkdir(parents=True, exist_ok=True)

SYSTEM = ("You are Meridian's analyst. Answer in one sentence: the figure or name, "
          "and nothing else.")

# In a real project these come from production runs a human approved — §23.10's
# feedback table joined to §25.6's runs. Here they are built from the golden set,
# which is the same shape and a much smaller sample.
cases = [c for c in load() if c.get("answer")]
examples = [{"messages": [
    {"role": "system", "content": SYSTEM},
    {"role": "user", "content": c["question"]},
    {"role": "assistant", "content": f"{c['answer']}."}]} | {"_id": c["id"]}
    for c in cases]

random.shuffle(examples)
split = int(len(examples) * 0.8)
train, hold = examples[:split], examples[split:]


def fingerprint(example: dict) -> str:
    user = next(m["content"] for m in example["messages"] if m["role"] == "user")
    return hashlib.sha256(user.strip().lower().encode()).hexdigest()[:16]


def validate(train: list[dict], hold: list[dict]) -> dict:
    problems = {}

    # 1. Leakage. The most common and most expensive mistake: a case in both sets
    #    makes the evaluation report memorisation as skill.
    overlap = {fingerprint(e) for e in train} & {fingerprint(e) for e in hold}
    problems["leaked between train and holdout"] = len(overlap)

    # 2. Duplicates within training. They do not add information and they quietly
    #    reweight whatever they duplicate.
    counts = Counter(fingerprint(e) for e in train)
    problems["duplicated inside training"] = sum(c - 1 for c in counts.values()
                                                 if c > 1)

    # 3. Shape. Every example must have the same roles in the same order, or the
    #    model learns the variation too.
    problems["wrong message shape"] = sum(
        1 for e in train
        if [m["role"] for m in e["messages"]] != ["system", "user", "assistant"])

    # 4. Balance. A training set that is 90% one kind of answer teaches that answer.
    kinds = Counter(e["_id"].split("-")[0] for e in train)
    top = kinds.most_common(1)[0]
    problems["most common family, share"] = round(top[1] / len(train), 2)
    return problems


report = validate(train, hold)
print(f"{len(examples)} examples: {len(train)} training, {len(hold)} holdout.\n")
for check, value in report.items():
    verdict = "ok" if (isinstance(value, int) and value == 0) else (
        "ok" if value <= 0.5 else "LOOK")
    print(f"  {check:<34}{value:>8}   {verdict}")

# Now break it on purpose, the way a careless split does.
careless = train + hold[:3]
broken = validate(careless, hold)
print(f"\n  the same checks after a careless split that reuses three holdout cases:\n")
for check, value in broken.items():
    verdict = "ok" if (isinstance(value, int) and value == 0) else (
        "ok" if isinstance(value, float) and value <= 0.5 else "LOOK")
    print(f"  {check:<34}{value:>8}   {verdict}")

for name, rows in (("train.jsonl", train), ("holdout.jsonl", hold)):
    (OUT / name).write_text("\n".join(
        json.dumps({k: v for k, v in r.items() if k != "_id"}) for r in rows))
json.dump({"total": len(examples), "train": len(train), "holdout": len(hold),
           "report": report, "broken": broken},
          open("code/30/_dataset.json", "w"), indent=1)
print(f"\n  written to {OUT}/train.jsonl and {OUT}/holdout.jsonl")

print()
print("Four checks, and the first one is the one that ruins projects.")
print()
print("A case in both sets makes your evaluation report memorisation as skill, and")
print("the report will look excellent. It is the same failure as testing on your")
print("training data in any other field, and it is easier to commit here because the")
print("'data' is prose and the duplicate is a rephrasing rather than a copy — which")
print("is why the check hashes a normalised question rather than the whole example.")
print()
print("On the size question: the honest answer is that hundreds of examples change")
print("behaviour and thousands are needed to change it reliably, and that the number")
print("matters far less than whether the examples agree with each other. Fifty")
print("consistent examples of a house style beat five hundred that were written by")
print("five people with different opinions — and an inconsistent training set is the")
print("one failure mode that more data makes worse.")
print()
print("What this listing does not do is run the fine-tune. That is a training job")
print("with a bill and a wait, and its result would be a model id that means nothing")
print("to a reader. §30.4 is the part that matters anyway: whatever comes back gets")
print("measured against the base model on Chapter 21's golden set, on cases the")
print("training never saw, and the comparison is the deliverable rather than the")
print("model.")
