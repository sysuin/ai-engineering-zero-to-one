# Clarity v0.3 against a real Meridian quarterly review.

import sys

sys.path.insert(0, "code")
from clarity.v0_3.extract import extract, weakest_region      # noqa: E402

from pathlib import Path                                       # noqa: E402

path = Path("data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md")
record = extract(path.read_text())

print(record.model_dump_json(indent=2))
print()
print(f"weakest region, computed in Python: {weakest_region(record).region.value}")
print(f"revenue there:                      ${weakest_region(record).revenue_usd:,.0f}")
