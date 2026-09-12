# skip
"""
A small searchable corpus for Chapter 11, built from Meridian's contracts.

Each contract has eight numbered clauses, and every clause becomes one chunk with its
heading intact — Chapter 10's structural chunking, reused. That gives ground truth for
free: a query about payment terms should return a clause 3 chunk.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, "code")
from clarity.v0_4.longdoc import chunk       # noqa: E402

CONTRACTS = Path("data/meridian/documents/contracts")

# What each numbered clause is about, so a returned chunk can be scored.
CLAUSE_TOPIC = {
    "1": "scope", "2": "price adjustment", "3": "payment",
    "4": "delivery and service levels", "5": "quality",
    "6": "liability", "7": "termination", "8": "governing law",
}


def load(limit: int = 12) -> list[dict]:
    """Every clause of the first `limit` contracts, as a flat list of chunks."""
    out = []
    for path in sorted(CONTRACTS.glob("*.md"))[:limit]:
        ref = path.stem.replace("contract-", "")
        for piece in chunk(path.read_text(), max_tokens=400):
            m = re.match(r"^(\d+)\.", piece.heading)
            out.append({
                "id": f"{ref}#{piece.heading[:22]}",
                "contract": ref,
                "clause": m.group(1) if m else None,
                "topic": CLAUSE_TOPIC.get(m.group(1)) if m else None,
                "text": piece.text,
            })
    return [c for c in out if c["clause"]]


# Ten questions, and the clause that answers each. Written by reading the contracts,
# deliberately using words the contracts mostly do not.
QUERIES = [
    ("How long do we have to settle an invoice?",            "3"),
    ("What is the ceiling on a price increase?",             "2"),
    ("Can we walk away from this agreement early?",          "7"),
    ("What if the goods turn up broken?",                    "5"),
    ("How much can we claim if they cause us a loss?",       "6"),
    ("Do they have to hit a delivery target?",               "4"),
    ("Which country's courts decide a dispute?",             "8"),
    ("What are they actually selling us?",                   "1"),
    ("How much warning before they charge us more?",         "2"),
    ("Is there a penalty for repeated late shipments?",      "4"),
]
