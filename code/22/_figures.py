# skip
"""Correct and wrong figures written the way people write them, with certain labels."""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clarity.evals.runner import load                           # noqa: E402


def build(seed: int = 22) -> list[dict]:
    rng = random.Random(seed)
    cases = [c for c in load() if c["kind"] in ("document", "warehouse") and c["answer"]
             and c["answer"].replace(",", "").replace(".", "").isdigit()
             and float(c["answer"].replace(",", "")) >= 100_000]
    rows = []
    for case in cases:
        value = float(case["answer"].replace(",", ""))
        wrong = value * rng.choice((0.88, 1.13))
        for label, figure in ((1, value), (0, wrong)):
            for form, text in (("millions", f"${figure / 1e6:.2f} million"),
                               ("thousands", f"about {figure / 1e3:,.0f} thousand")):
                rows.append(dict(id=case["id"], form=form, label=label,
                                 question=case["question"], expected=case["answer"],
                                 answer=f"It was {text}.", case=case))
    return rows
