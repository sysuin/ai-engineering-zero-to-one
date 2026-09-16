# timeout: 900
# A ticket is one of several kinds, and each kind has different fields. A discriminated
# union lets the schema say so, instead of one record full of nulls.

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal, Union

from openai import OpenAI
from pydantic import BaseModel, Field

from clarity.config import MODEL_FAST

client = OpenAI()
rows = [json.loads(line) for line in
        Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()][:150]


class Delivery(BaseModel):
    kind: Literal["Delivery"]
    order_number: str | None = Field(description="Digits only, if the ticket gives one.")


class Billing(BaseModel):
    kind: Literal["Billing"]
    invoice_number: str | None = Field(description="Like INV-12345, if the ticket gives one.")


class Quality(BaseModel):
    kind: Literal["Quality"]
    sku: str | None = Field(description="Like MRD-CLE-001, if the ticket gives one.")


class Returns(BaseModel):
    kind: Literal["Returns"]


class Account(BaseModel):
    kind: Literal["Account"]


class Ticket(BaseModel):
    # Pydantic can mark the union as discriminated by `kind`, which it writes as `oneOf`.
    # Strict mode refuses `oneOf`; a plain union becomes `anyOf`, which it accepts, and the
    # Literal in each branch still tells both the model and the validator which is which.
    issue: Union[Delivery, Billing, Quality, Returns, Account]


def extract(row: dict) -> Ticket | None:
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, response_format=Ticket, max_completion_tokens=120,
        messages=[{"role": "system", "content": "Classify the support ticket and extract the "
                                                "identifier its kind calls for."},
                  {"role": "user", "content": row["body"]}],
    ).choices[0].message.parsed


with ThreadPoolExecutor(max_workers=12) as pool:
    parsed = list(pool.map(extract, rows))

kind_right = sum(p is not None and p.issue.kind == r["category"] for p, r in zip(parsed, rows))
print(f"{len(rows)} tickets; kind matches the label on {kind_right} ({kind_right / len(rows):.0%})\n")

PATTERNS = {"Delivery": ("order_number", r"order\s+#?(\d+)"),
            "Billing": ("invoice_number", r"(INV-\d+)"),
            "Quality": ("sku", r"(MRD-[A-Z]{3}-\d{3})")}
print(f"  {'kind':9} {'tickets':>7} {'id in text':>11} {'id extracted':>13} {'wrong id':>9}")
for kind, (field, pattern) in PATTERNS.items():
    group = [(p, r) for p, r in zip(parsed, rows) if p is not None and p.issue.kind == kind]
    in_text = extracted = wrong = 0
    for p, r in group:
        m = re.search(pattern, r["body"], re.I)
        value = getattr(p.issue, field)
        if m:
            in_text += 1
            extracted += value is not None and value.upper() == m.group(1).upper()
        if value is not None and (not m or value.upper() != m.group(1).upper()):
            wrong += 1
    print(f"  {kind:9} {len(group):>7} {in_text:>11} {extracted:>13} {wrong:>9}")

print("\nExample:", parsed[0].model_dump() if parsed[0] else None)
