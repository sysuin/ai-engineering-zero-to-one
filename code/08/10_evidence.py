# timeout: 900
# Provenance: ask for the sentence each value came from, then check that the sentence is
# really in the document. A value with a quote you can find is a value you can audit.

import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI
from pydantic import BaseModel, Field

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()


class Sourced(BaseModel):
    value: int
    quote: str = Field(description="The exact sentence from the agreement that states the value.")


class Terms(BaseModel):
    payment_days: Sourced
    price_cap_pct: Sourced
    price_notice_days: Sourced


def normalise(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().strip('"“”').lower()


def extract(c: dict):
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, response_format=Terms, max_completion_tokens=500,
        messages=[{"role": "user", "content": f"Extract these terms.\n\n<contract>\n{c['text']}\n</contract>"}],
    ).choices[0].message.parsed


with ThreadPoolExecutor(max_workers=12) as pool:
    records = list(pool.map(extract, contracts))

truth_key = {"payment_days": "payment_days", "price_cap_pct": "cap_pct",
             "price_notice_days": "notice_days"}
print(f"  {'field':18} {'value right':>11} {'quote found':>12} {'quote contains value':>21}")
for field, key in truth_key.items():
    right = found = contains = 0
    for record, c in zip(records, contracts):
        sourced = getattr(record, field)
        right += sourced.value == c[key]
        found += normalise(sourced.quote) in normalise(c["text"])
        contains += str(sourced.value) in sourced.quote
    n = len(contracts)
    print(f"  {field:18} {right:>6}/{n:<4} {found:>7}/{n:<4} {contains:>16}/{n:<4}")

print("\nA quote that is not found verbatim is not necessarily invented — it may be lightly")
print("reworded — but it cannot be audited, which is the property the check is buying.")
