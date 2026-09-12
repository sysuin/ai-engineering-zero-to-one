# timeout: 900
# Describe the shape once, in Python, and let everything else follow from it.

import json
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()


class PaymentBasis(str, Enum):
    """The model must choose one of these. It cannot invent a sixth option."""
    INVOICE = "invoice"
    RECEIPT = "receipt"


class ContractTerms(BaseModel):
    """One supplier agreement, as the fields Meridian actually cares about."""

    supplier: str = Field(description="The Supplier's legal name, exactly as written.")
    payment_days: int = Field(description="Days allowed to pay a valid invoice.")
    payment_basis: PaymentBasis = Field(
        description="Whether the payment clock starts at the invoice date or at receipt.")
    price_cap_pct: int = Field(
        description="Maximum percentage price rise permitted in twelve months.")
    price_notice_days: int = Field(
        description="Days of written notice required before a price rise.")
    liability_cap_usd: int | None = Field(
        description="Any flat dollar liability cap. Null if the contract states only "
                    "a percentage, or no cap at all.")


def extract(text: str) -> ContractTerms | None:
    response = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        response_format=ContractTerms,
        messages=[
            {"role": "system", "content": "Extract contract terms exactly as written. "
                                          "Never guess a value that is not stated."},
            {"role": "user", "content": f"<contract>\n{text}\n</contract>"},
        ],
    )
    return response.choices[0].message.parsed


with ThreadPoolExecutor(max_workers=12) as pool:
    records = list(pool.map(lambda c: extract(c["text"]), contracts))

print("The schema, as the API sees it:")
print(json.dumps(ContractTerms.model_json_schema()["properties"]["payment_basis"],
                 indent=2))

print(f"\nExtracted {sum(r is not None for r in records)} of {len(contracts)} contracts.")
print(f"Every one is a {type(records[0]).__name__}, not a string.\n")
print(records[0].model_dump_json(indent=2))

# Accuracy against ground truth, field by field.
print("\nField-level accuracy against the generated ground truth")
for field, key in (("payment_days", "payment_days"),
                   ("price_cap_pct", "cap_pct"),
                   ("price_notice_days", "notice_days")):
    hits = sum(getattr(r, field) == c[key] for r, c in zip(records, contracts))
    print(f"  {field:20} {hits}/{len(contracts)}  {hits / len(contracts):.0%}")

basis_hits = sum(r.payment_basis.value in c["payment_code"].lower() or
                 (r.payment_basis is PaymentBasis.INVOICE and "INV" in c["payment_code"])
                 or (r.payment_basis is PaymentBasis.RECEIPT and "RCP" in c["payment_code"])
                 for r, c in zip(records, contracts))
print(f"  {'payment_basis':20} {basis_hits}/{len(contracts)}  "
      f"{basis_hits / len(contracts):.0%}")

absent = sum(r.liability_cap_usd is None for r in records)
print(f"\n  liability_cap_usd is null in {absent} of {len(contracts)} — the contracts")
print(f"  that state only a percentage. A field that is allowed to be absent is how a")
print(f"  model says 'not in the document' without inventing something.")
