# The JSON Schema underneath a Pydantic model, before and after it is made strict — and
# three schema features the provider may or may not honour, found by asking.

import json
from enum import Enum

import openai
from openai import OpenAI
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field

from clarity.config import MODEL_FAST


class PaymentBasis(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"


class Terms(BaseModel):
    payment_days: int = Field(description="Days allowed to pay a valid invoice.")
    payment_basis: PaymentBasis
    liability_cap_usd: int | None = Field(description="Flat cap in dollars, or null.")


print("What Pydantic writes:")
print(json.dumps(Terms.model_json_schema(), indent=1))
print("\nWhat strict mode sends (differences from the above):")
strict = to_strict_json_schema(Terms)
print("  required:             ", strict["required"])
print("  additionalProperties: ", strict["additionalProperties"])
print("  liability_cap_usd:    ", json.dumps(strict["properties"]["liability_cap_usd"]))

client = OpenAI()
PROBES = {
    "minimum / maximum": {"type": "integer", "minimum": 1, "maximum": 365},
    "pattern":           {"type": "string", "pattern": "^MSC-\\d{4}-\\d{3}$"},
    "maxItems":          {"type": "array", "items": {"type": "string"}, "maxItems": 2},
}
print("\nDoes the provider accept these keywords in a strict schema?")
for name, spec in PROBES.items():
    schema = {"type": "object", "properties": {"value": spec}, "required": ["value"],
              "additionalProperties": False}
    try:
        client.chat.completions.create(
            model=MODEL_FAST, max_completion_tokens=60,
            messages=[{"role": "user", "content": "Give any valid value."}],
            response_format={"type": "json_schema", "json_schema": {
                "name": "probe", "strict": True, "schema": schema}})
        print(f"  {name:18} accepted")
    except openai.BadRequestError as error:
        print(f"  {name:18} rejected: {str(error.message)[:60]}")
print("\nWhatever the answer, Pydantic enforces its own constraints on the way back in.")
