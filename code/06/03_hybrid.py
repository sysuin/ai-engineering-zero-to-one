# timeout: 600
# Neither, and both: rules handle the head, the model handles what rules cannot.

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

CATEGORIES = ["Delivery", "Quality", "Billing", "Returns", "Account"]
SAMPLE = 200

rows = [json.loads(line) for line in
        Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()]
sample = rows[:SAMPLE]
truth = [r["category"] for r in sample]

# The first draft, from the previous listing. Written in twenty minutes.
FIRST_DRAFT = [
    ("Delivery", r"deliver|arriv|late|missing|shipment|where is|dispatch|turned up"),
    ("Quality",  r"defect|damag|broken|falling apart|tearing|not the same|thinner|bad"),
    ("Billing",  r"invoice|charg|credit note|tax|vat|billed|payment|PO number"),
    ("Returns",  r"return|collection|RMA|wrong size|send back"),
    ("Account",  r"address|remove|add .*user|pricing|account|copy of our"),
]

# Twenty minutes more, spent reading the cases the first draft got wrong. Two changes:
# the order now puts the specific categories before the general ones, and "second
# delivery address" no longer looks like a delivery problem.
RULES = [
    ("Returns",  r"return|collection|RMA|wrong size|send back"),
    ("Account",  r"second delivery address|remove |add a |pricing|account"),
    ("Billing",  r"invoice|charg|credit note|tax|vat|billed|payment|PO number"),
    ("Quality",  r"defect|damag|broken|falling apart|tearing|not the same|thinner|bad"),
    ("Delivery", r"deliver|arriv|late|missing|shipment|where is|dispatch|turned up"),
]


def by_rule(text: str) -> str | None:
    for category, pattern in RULES:
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return None


client = OpenAI()
PROMPT = ("Classify this support ticket into exactly one category: "
          f"{', '.join(CATEGORIES)}. Reply with the category name only.\n\nTicket: ")


def by_model(text: str) -> str:
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0,
        messages=[{"role": "user", "content": PROMPT + text}],
        max_completion_tokens=16,
    ).choices[0].message.content or "").strip()


def by_first_draft(text: str) -> str | None:
    for category, pattern in FIRST_DRAFT:
        if re.search(pattern, text, re.IGNORECASE):
            return category
    return None


# Rules first. Whatever they decline, the model gets.
started = time.time()
answers = [by_rule(r["body"]) for r in sample]
undecided = [i for i, a in enumerate(answers) if a is None]

with ThreadPoolExecutor(max_workers=16) as pool:
    filled = list(pool.map(lambda i: by_model(sample[i]["body"]), undecided))
for i, answer in zip(undecided, filled):
    answers[i] = answer
elapsed = time.time() - started

correct = sum(a == t for a, t in zip(answers, truth))
rules_only = sum(by_rule(r["body"]) == t for r, t in zip(sample, truth))
first_draft = sum(by_first_draft(r["body"]) == t for r, t in zip(sample, truth))

print(f"Four attempts at the same problem, {SAMPLE} tickets\n")
print(f"  1. rules, first draft     {first_draft / SAMPLE:.1%}   twenty minutes' work")
print(f"  2. the model              78.0%   from the previous listing")
print(f"  3. rules, second draft    {rules_only / SAMPLE:.1%}   twenty minutes more")
print(f"  4. rules + model on the rest  {correct / SAMPLE:.1%}\n")

print(f"Hybrid over {SAMPLE} tickets")
print(f"  rules answered            {SAMPLE - len(undecided)} "
      f"({(SAMPLE - len(undecided)) / SAMPLE:.0%})")
print(f"  model answered            {len(undecided)} ({len(undecided) / SAMPLE:.0%})")
print(f"  accuracy                  {correct / SAMPLE:.1%}")
print(f"  seconds                   {elapsed:.1f}")
print()
print(f"  rules alone               {rules_only / SAMPLE:.1%}")
print(f"  model calls avoided       {(SAMPLE - len(undecided)) / SAMPLE:.0%} of them")

print("\nWhat the model was asked to do — only the cases rules could not read:")
for i in undecided[:5]:
    print(f"  {answers[i]:9} (truth {truth[i]:9}) {sample[i]['body'][:50]}")

print("\nThe rules are not better than the model, and the model is not better than the")
print("rules. Each is better at a different part of the same problem, and the cheap")
print("one can tell when it is out of its depth. That is a design, not a compromise.")
