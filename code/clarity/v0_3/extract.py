"""
Clarity v0.3 — turn a Meridian quarterly review into a validated record.

v0.1 produced a paragraph a person had to read. This produces an object a program can
use: typed fields, an enum the model cannot escape, optional fields for things the
document may not say, and — the part that matters — no field the model is allowed to
have an opinion about.
"""
from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST      # noqa: E402
from clarity.prompts import load           # noqa: E402


class Region(str, Enum):
    NORTHEAST = "Northeast"
    SOUTHEAST = "Southeast"
    MIDWEST = "Midwest"
    WEST = "West"
    SOUTHWEST = "Southwest"


class RegionResult(BaseModel):
    region: Region
    revenue_usd: float = Field(ge=0)


class QuarterlyReview(BaseModel):
    """
    What Clarity extracts from one quarterly business review.

    Note what is absent. There is no `performance` field, no `outlook_sentiment`, no
    `is_concerning`. Every one of those would be the model's opinion wearing a typed
    field's clothing, and Clarity's policy about what counts as a bad quarter belongs
    in code that can be tested and shown to someone.
    """

    year: int = Field(ge=2000, le=2100)
    quarter: int = Field(ge=1, le=4)

    revenue_usd: float = Field(ge=0, description="Total revenue stated for the quarter.")
    orders: int = Field(ge=0)
    gross_margin_pct: float = Field(ge=-100, le=100)
    units_shipped: int = Field(ge=0)

    by_region: list[RegionResult] = Field(
        description="Revenue for every region the document reports.")

    account_loss: str | None = Field(
        default=None,
        description="If the document reports losing a named account, the account's "
                    "name. Null if it does not.")
    stated_cause: str | None = Field(
        default=None,
        description="The explanation the document itself gives for the largest "
                    "movement, quoted or closely paraphrased. Null if none is given.")

    @field_validator("by_region")
    @classmethod
    def regions_are_distinct(cls, value: list[RegionResult]) -> list[RegionResult]:
        names = [r.region for r in value]
        if len(names) != len(set(names)):
            raise ValueError("the same region appears twice")
        return value


def extract(document: str, client: OpenAI | None = None) -> QuarterlyReview:
    """Extract one review. Raises if the model returns something unusable."""
    client = client or OpenAI()
    response = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=1200,
        response_format=QuarterlyReview,
        messages=[
            {"role": "system", "content": load("extract_review")},
            {"role": "user", "content": f"<document>\n{document}\n</document>"},
        ],
    )
    choice = response.choices[0]
    if choice.finish_reason == "length":
        raise ValueError("truncated before the record was complete")
    if choice.message.parsed is None:
        raise ValueError(f"no parsed record: {choice.message.refusal or 'unknown'}")
    return choice.message.parsed


def weakest_region(review: QuarterlyReview) -> RegionResult:
    """Computed, not asked for. Clarity never lets the model rank anything."""
    return min(review.by_region, key=lambda r: r.revenue_usd)


if __name__ == "__main__":
    path = Path(sys.argv[1] if len(sys.argv) > 1 else
                "data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md")
    record = extract(path.read_text())
    print(record.model_dump_json(indent=2))
    print(f"\nweakest region (computed in Python): "
          f"{weakest_region(record).region.value}")
