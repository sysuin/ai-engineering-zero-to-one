# timeout: 900
# The model's own probability was not a usable confidence. Three other signals for the same
# 200 tickets — agreement with the rules, with a second sample, with a reworded prompt — and
# what each is worth when the unsure tickets go to a person.

import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

CATEGORIES = ["Delivery", "Quality", "Billing", "Returns", "Account"]
rows = [json.loads(line) for line in
        Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()][:200]
client = OpenAI()
REVIEW_COST, ERROR_COST = 0.99, 6.00          # as in 05_threshold.py

RULES = [                                     # the second draft, from 03_hybrid.py
    ("Returns",  r"return|collection|RMA|wrong size|send back"),
    ("Account",  r"second delivery address|remove |add a |pricing|account"),
    ("Billing",  r"invoice|charg|credit note|tax|vat|billed|payment|PO number"),
    ("Quality",  r"defect|damag|broken|falling apart|tearing|not the same|thinner|bad"),
    ("Delivery", r"deliver|arriv|late|missing|shipment|where is|dispatch|turned up"),
]
PROMPT = ("Classify this support ticket into exactly one category: "
          f"{', '.join(CATEGORIES)}. Reply with the category name only.\n\nTicket: ")
REWORDED = ("A customer wrote to us. Which team should handle it — Delivery, Quality, Billing, Returns "
            "or Account? Answer with the team's name and nothing else.\n\nMessage: ")


def rule(text: str) -> str | None:
    return next((c for c, p in RULES if re.search(p, text, re.IGNORECASE)), None)


def ask(prompt: str, text: str, temperature: float = 0.0) -> tuple[str, float]:
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=temperature, logprobs=True, max_completion_tokens=8,
        messages=[{"role": "user", "content": prompt + text}])
    return ((response.choices[0].message.content or "").strip(),
            math.exp(response.choices[0].logprobs.content[0].logprob))


def signals(row: dict) -> dict:
    answer, probability = ask(PROMPT, row["body"])
    return {"right": answer == row["category"],
            "probability ≥ 0.999": probability >= 0.999,
            "rules agree": rule(row["body"]) == answer,
            "a second sample agrees": ask(PROMPT, row["body"], temperature=1.0)[0] == answer,
            "a reworded prompt agrees": ask(REWORDED, row["body"])[0] == answer}


with ThreadPoolExecutor(max_workers=12) as pool:
    results = list(pool.map(signals, rows))
for r in results:
    r["rules and reworded both agree"] = r["rules agree"] and r["a reworded prompt agrees"]

n = len(results)
print(f"{n} tickets; the model alone is right on {sum(r['right'] for r in results)}\n")
print(f"  {'confident when':30}{'automated':>10}{'right when confident':>22}{'right when unsure':>19}{'cost / 1,000':>14}")
for signal in ["probability ≥ 0.999", "rules agree", "a second sample agrees",
               "a reworded prompt agrees", "rules and reworded both agree"]:
    sure = [r for r in results if r[signal]]
    unsure = [r for r in results if not r[signal]]
    cost = (len(unsure) * REVIEW_COST + sum(not r["right"] for r in sure) * ERROR_COST) / n * 1000
    right_sure = f"{sum(r['right'] for r in sure) / len(sure):.1%}" if sure else "—"
    right_unsure = f"{sum(r['right'] for r in unsure) / len(unsure):.1%}" if unsure else "—"
    print(f"  {signal:30}{len(sure) / n:>10.0%}{right_sure:>22}{right_unsure:>19}{'$' + f'{cost:,.0f}':>14}")
all_auto = sum(not r["right"] for r in results) * ERROR_COST / n * 1000
print(f"\n  for comparison: automate everything ${all_auto:,.0f}, review everything ${REVIEW_COST * 1000:,.0f}")
