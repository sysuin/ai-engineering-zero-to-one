# Validation is not optional, and neither is what you do when it fails.
#
# A strict schema guarantees the shape. It guarantees nothing about the values being
# sensible, and it cannot help at all when the response is cut off.

import json

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contract = load()[0]


class Terms(BaseModel):
    payment_days: int = Field(ge=1, le=365)
    price_cap_pct: int = Field(ge=0, le=100)

    @field_validator("payment_days")
    @classmethod
    def sensible_payment_terms(cls, value: int) -> int:
        # Business rules the schema cannot express. Meridian has never agreed a term
        # longer than 120 days, so a larger number is a misread, not a discovery.
        if value > 120:
            raise ValueError(f"{value} days is outside anything Meridian agrees to")
        return value


def attempt(messages: list[dict], budget: int) -> tuple[str | None, str]:
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=budget,
        response_format={"type": "json_schema", "json_schema": {
            "name": "terms", "strict": True,
            "schema": {"type": "object",
                       "properties": {"payment_days": {"type": "integer"},
                                      "price_cap_pct": {"type": "integer"}},
                       "required": ["payment_days", "price_cap_pct"],
                       "additionalProperties": False}}},
        messages=messages,
    )
    return response.choices[0].message.content, response.choices[0].finish_reason


def extract_with_repair(text: str, budget: int, attempts: int = 3):
    messages = [{"role": "user", "content": f"Extract the terms.\n\n{text}"}]
    for attempt_no in range(1, attempts + 1):
        raw, finish = attempt(messages, budget)
        print(f"  attempt {attempt_no}: finish_reason={finish}, "
              f"{len(raw or '')} chars")
        try:
            if finish == "length":
                raise ValueError("response was truncated before it finished")
            return Terms.model_validate_json(raw), attempt_no
        except (ValidationError, ValueError, TypeError) as error:
            reason = str(error).split("\n")[0][:70]
            print(f"             rejected: {reason}")
            if attempt_no == attempts:
                return None, attempt_no
            # Tell it what was wrong. A retry that repeats the request repeats the answer.
            messages += [
                {"role": "assistant", "content": raw or ""},
                {"role": "user", "content": f"That was rejected: {reason}. "
                                            "Return corrected JSON."},
            ]
            budget = max(budget * 4, 200)


print("With a budget too small to finish the object:")
record, tries = extract_with_repair(contract["text"], budget=16)
print(f"  -> {record}  after {tries} attempt(s)\n")

print("With a sensible budget:")
record, tries = extract_with_repair(contract["text"], budget=200)
print(f"  -> {record}  after {tries} attempt(s)")

print()
print("Three layers, and each catches what the one below cannot:")
print("  the strict schema   guarantees the shape")
print("  Pydantic bounds     guarantee the values are in range")
print("  a field validator   guarantees they are sensible for your business")
print()
print("None of them helps with truncation, which arrives as valid-looking nothing.")
print("finish_reason is the only signal, exactly as Chapter 4 warned.")
