# The leakage check in 02 hashes the question, lower-cased. A holdout question that differs
# from a training question only in its quarter passes that check and tests nothing new.
# Reads the split 02 wrote.

import json
import random
import re
from collections import defaultdict
from pathlib import Path

DATA = Path("data/meridian/finetune")
train = [json.loads(line) for line in (DATA / "train.jsonl").open()]
hold = [json.loads(line) for line in (DATA / "holdout.jsonl").open()]


def question(example: dict) -> str:
    return next(m["content"] for m in example["messages"] if m["role"] == "user")


def template(text: str) -> str:
    t = re.sub(r"\b20\d\d\b", "<year>", text)
    t = re.sub(r"\bQ[1-4]\b", "<quarter>", t)
    t = re.sub(r"MSC-\d{4}-\d{3}", "<contract>", t)
    return re.sub(r"\d[\d,.]*", "<n>", t.lower())


seen = {template(question(e)) for e in train}
same_shape = [e for e in hold if template(question(e)) in seen]
print(f"holdout: {len(hold)} examples")
print(f"  exact duplicates of a training question      "
      f"{sum(question(e).strip().lower() in {question(t).strip().lower() for t in train} for e in hold)}")
print(f"  same template as a training question         {len(same_shape)}")
for e in same_shape[:3]:
    print(f"    {question(e)}")

# Split by template instead: every question of a shape goes to one side or the other.
examples = train + hold
groups = defaultdict(list)
for e in examples:
    groups[template(question(e))].append(e)
keys = sorted(groups)
random.Random(30).shuffle(keys)
held_keys, count = set(), 0
for key in keys:
    if count >= len(hold):
        break
    held_keys.add(key)
    count += len(groups[key])
new_hold = [e for k in held_keys for e in groups[k]]
new_train = [e for k in keys if k not in held_keys for e in groups[k]]
overlap = {template(question(e)) for e in new_train} & {template(question(e)) for e in new_hold}
print(f"\nsplit by template: {len(new_train)} training, {len(new_hold)} holdout, "
      f"{len(overlap)} templates on both sides")
