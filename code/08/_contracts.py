# skip
"""
Shared setup for Chapter 7: forty supplier contracts, with ground truth.

The task is deliberately one where a model is the right tool. Chapter 6 showed that
keyword rules beat a model at classifying tickets; here the answer is a number buried in
one clause of a forty-clause document, phrased two different ways, and rules do badly.

Ground truth is extracted with a regular expression from the generated text — which works
only because we generated it. On a real corpus this is the part a person has to do, and
Chapter 21 is about doing it properly.
"""
from __future__ import annotations

import re
from pathlib import Path

CONTRACTS = Path("data/meridian/documents/contracts")

CAP = re.compile(r"no more than\s+(\d+)%|adjustment of up to\s+(\d+)%")
NOTICE = re.compile(r"on not less than (\d+) days|subject to (\d+) days")
PAYMENT = re.compile(r"within (\d+) days of (the date of a valid invoice|receipt)")


def load() -> list[dict]:
    """Every contract, with the three answers we will be asking for."""
    out = []
    for path in sorted(CONTRACTS.glob("*.md")):
        text = path.read_text()
        cap = CAP.search(text)
        notice = NOTICE.search(text)
        payment = PAYMENT.search(text)
        out.append({
            "ref": path.stem.replace("contract-", ""),
            "text": text,
            "cap_pct": int(cap.group(1) or cap.group(2)),
            "notice_days": int(notice.group(1) or notice.group(2)),
            "payment_days": int(payment.group(1)),
            # Meridian's internal shorthand for a payment term. A model cannot guess
            # this convention; it has to be shown. That makes it a fair test of
            # few-shot prompting.
            "payment_code": f"NET{int(payment.group(1))}-"
                            f"{'INV' if 'invoice' in payment.group(2) else 'RCP'}",
        })
    return out


def score(answers: list[str | None], truth: list[int]) -> float:
    """Fraction correct, where an answer counts if the right number is in it."""
    hits = 0
    for answer, expected in zip(answers, truth):
        if answer is None:
            continue
        digits = re.findall(r"\d+", str(answer))
        if digits and int(digits[0]) == expected:
            hits += 1
    return hits / len(truth)
