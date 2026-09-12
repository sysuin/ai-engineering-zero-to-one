# timeout: 600
# The chapter's central claim, measured rather than asserted:
# how good is a rule, really, and is a model worth what it costs?

import json
import re
import time
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

CATEGORIES = ["Delivery", "Quality", "Billing", "Returns", "Account"]
SAMPLE = 200

rows = [json.loads(line) for line in
        Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()]
sample = rows[:SAMPLE]

# ------------------------------------------------------------------ the rules
# Fourteen keywords, written in about twenty minutes by reading fifty tickets.
RULES = [
    ("Delivery", r"deliver|arriv|late|missing|shipment|where is|dispatch|turned up"),
    ("Quality",  r"defect|damag|broken|falling apart|tearing|not the same|thinner|bad"),
    ("Billing",  r"invoice|charg|credit note|tax|vat|billed|payment|PO number"),
    ("Returns",  r"return|collection|RMA|wrong size|send back"),
    ("Account",  r"address|remove|add .*user|pricing|account|copy of our"),
]


def classify_by_rule(text: str) -> str | None:
    for category, pattern in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return None


# ------------------------------------------------------------------ the model
client = OpenAI()
PROMPT = ("Classify this support ticket into exactly one category: "
          f"{', '.join(CATEGORIES)}. Reply with the category name only.\n\nTicket: ")


def classify_by_model(text: str) -> str:
    answer = client.chat.completions.create(
        model=MODEL_FAST, temperature=0,
        messages=[{"role": "user", "content": PROMPT + text}],
        max_completion_tokens=16,
    )
    return (answer.choices[0].message.content or "").strip()


# ------------------------------------------------------------------ measure both
started = time.time()
rule_answers = [classify_by_rule(r["body"]) for r in sample]
rule_seconds = time.time() - started

# Two hundred sequential calls take minutes; the same calls in parallel take seconds.
# This is Chapter 25's lesson arriving early, because the alternative is waiting.
from concurrent.futures import ThreadPoolExecutor       # noqa: E402

started = time.time()
with ThreadPoolExecutor(max_workers=16) as pool:
    model_answers = list(pool.map(lambda r: classify_by_model(r["body"]), sample))
model_seconds = time.time() - started

truth = [r["category"] for r in sample]
rule_right = sum(a == t for a, t in zip(rule_answers, truth))
rule_abstained = sum(a is None for a in rule_answers)
model_right = sum(a == t for a, t in zip(model_answers, truth))

print(f"{SAMPLE} tickets, ground truth from the dataset\n")
print(f"{'':10} {'correct':>9} {'accuracy':>9} {'no answer':>10} {'seconds':>9}")
print(f"{'rules':10} {rule_right:>9} {rule_right / SAMPLE:>8.1%} "
      f"{rule_abstained:>10} {rule_seconds:>9.3f}")
print(f"{'model':10} {model_right:>9} {model_right / SAMPLE:>8.1%} "
      f"{'0':>10} {model_seconds:>9.1f}")

# Where does each one fail?
print("\nWhere the rules fail")
misses = [(r, a, t) for r, a, t in zip(sample, rule_answers, truth) if a != t][:6]
for row, answer, true_label in misses:
    print(f"  said {str(answer):9} truth {true_label:9} {row['body'][:52]}")

print("\nWhere the model fails")
misses = [(r, a, t) for r, a, t in zip(sample, model_answers, truth) if a != t][:6]
for row, answer, true_label in misses:
    print(f"  said {answer:9} truth {true_label:9} {row['body'][:52]}")

Path("code/06/_bake_off.json").write_text(json.dumps({
    "sample": SAMPLE,
    "rules": {"correct": rule_right, "abstained": rule_abstained,
              "seconds": round(rule_seconds, 4)},
    "model": {"correct": model_right, "seconds": round(model_seconds, 2)},
    "rule_answers": rule_answers, "model_answers": model_answers, "truth": truth,
}))
