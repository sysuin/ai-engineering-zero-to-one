# timeout: 900
# The model's ticket errors in Chapter 6 were mostly disagreements about what a category means.
# Say what each category means — the organisation's convention, not the dictionary's — and measure.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
everything = [json.loads(line) for line in
              Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()]
rows, held_out = everything[:200], everything[200:]      # the definitions were written after
                                                         # reading the errors on the first 200
BASE = ("Classify this support ticket into exactly one category: Delivery, Quality, Billing, "
        "Returns, Account. Reply with the category name only.")
DEFINITIONS = """
What the categories mean at Meridian:
- Delivery: an order that is late, missing, or partly missing in transit.
- Quality: a product that is damaged, defective or not as specified.
- Billing: invoices, charges, credit notes, tax and purchase order numbers.
- Returns: sending goods back, including booking or missing a collection.
- Account: changes to the customer's account, users, addresses or price lists.
A ticket may be written in any language; classify it by what it is about."""
PROMPTS = {"categories named only": BASE,
           "with one-line definitions": BASE + "\n" + DEFINITIONS}


def classify(prompt: str, row: dict) -> str:
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=30,
        messages=[{"role": "user", "content": f"{prompt}\n\nTicket: {row['body']}"}],
    ).choices[0].message.content or "").strip().strip(".")


with ThreadPoolExecutor(max_workers=16) as pool:
    answers = {name: list(pool.map(lambda r, p=p: classify(p, r), rows)) for name, p in PROMPTS.items()}
    held = {name: list(pool.map(lambda r, p=p: classify(p, r), held_out)) for name, p in PROMPTS.items()}


def accuracy(got, truth_rows):
    return sum(g == r["category"] for g, r in zip(got, truth_rows)) / len(truth_rows)


print(f"  {'':28}{'the 200 whose errors were read':>32}{'400 tickets not looked at':>28}")
for name in PROMPTS:
    print(f"  {name:28}{accuracy(answers[name], rows):>32.1%}{accuracy(held[name], held_out):>28.1%}")

before, after = answers["categories named only"], answers["with one-line definitions"]
fixed = [(r, b) for r, b, a in zip(rows, before, after) if b != r["category"] and a == r["category"]]
broken = [(r, a) for r, b, a in zip(rows, before, after) if b == r["category"] and a != r["category"]]
still = [(r, a) for r, b, a in zip(rows, before, after) if b != r["category"] and a != r["category"]]
print(f"\non the first 200: fixed by the definitions {len(fixed)}, broken by them {len(broken)}, "
      f"wrong either way {len(still)}")
for label, items in (("fixed", fixed), ("broken", broken), ("still wrong", still)):
    seen = set()
    for r, answer in items:
        key = r["body"][:12]
        if key not in seen:
            seen.add(key)
            print(f"  {label:12} {r['body'][:46]:47} said {answer:9} truth {r['category']}")
