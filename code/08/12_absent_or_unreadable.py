# timeout: 1200
# The same field, when the document states it, when it does not, and when it is there but
# cannot be read. Three schemas: a required integer, a nullable one, and a status beside it.

import random
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
SECTION = re.compile(r"## 3\. Payment.*?(?=\n## )", re.S)


def garble(text: str, seed: int) -> str:
    """What a bad scan leaves: the heading survives, the clause is noise, every digit is lost."""
    rng = random.Random(seed)
    return "".join("□" if ch.isdigit() else rng.choice("#~?@%") if ch.isalpha() and rng.random() < 0.45
                   else ch for ch in text)


def variant(contract: dict, condition: str) -> str:
    text = contract["text"]
    if condition == "absent":
        return SECTION.sub("", text)
    if condition == "unreadable":
        return SECTION.sub(lambda m: "## 3. Payment\n" + garble(m.group(0)[len("## 3. Payment"):],
                                                                 hash(contract["ref"]) % 1000), text)
    return text


class Required(BaseModel):
    payment_days: int = Field(description="Days allowed to pay a valid invoice.")


class Nullable(BaseModel):
    payment_days: int | None = Field(description="Days allowed to pay a valid invoice. Null if not stated.")


class WithStatus(BaseModel):
    status: Literal["stated", "not_stated", "unreadable"] = Field(
        description="stated: the agreement gives the payment term. not_stated: it has no payment clause. "
                    "unreadable: there is a payment clause but its text cannot be read.")
    payment_days: int | None = Field(description="Days allowed to pay, if stated and readable; else null.")


SCHEMAS = {"required integer": Required, "nullable integer": Nullable, "status and nullable": WithStatus}


def extract(contract: dict, condition: str, schema) -> BaseModel | None:
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200, response_format=schema,
        messages=[{"role": "user", "content": "Extract the payment term from this agreement.\n\n"
                   f"<contract>\n{variant(contract, condition)}\n</contract>"}]).choices[0].message.parsed


CONDITIONS = ("stated", "absent", "unreadable")
jobs = [(c, cond, name) for name in SCHEMAS for cond in CONDITIONS for c in contracts]
with ThreadPoolExecutor(max_workers=16) as pool:
    results = list(pool.map(lambda j: (j, extract(j[0], j[1], SCHEMAS[j[2]])), jobs))

n = len(contracts)
print(f"{n} contracts; the payment clause left alone, removed, or garbled like a bad scan\n")
print(f"  {'schema':22}{'condition':12}{'a number':>9}{'the true one':>14}{'null':>6}{'status right':>14}")
for name in SCHEMAS:
    for cond in CONDITIONS:
        mine = [(c, r) for (c, k, s), r in results if s == name and k == cond]
        numbers = sum(r is not None and r.payment_days is not None for _, r in mine)
        right = sum(r is not None and r.payment_days == c["payment_days"] for c, r in mine)
        nulls = sum(r is not None and r.payment_days is None for _, r in mine)
        want = {"stated": "stated", "absent": "not_stated", "unreadable": "unreadable"}[cond]
        status = (f"{sum(r is not None and r.status == want for _, r in mine):>11}/{n}"
                  if name == "status and nullable" else f"{'—':>14}")
        print(f"  {name:22}{cond:12}{numbers:>9}{right:>14}{nulls:>6}{status}")

sample = variant(contracts[0], "unreadable")
print("\nwhen the clause is absent or garbled, 'the true one' is a lucky guess: no other clause states it")
print("\nthe garbled clause, as the model saw it:\n  " +
      " ".join(SECTION.search(sample).group(0).split())[:150] if SECTION.search(sample) else "")
