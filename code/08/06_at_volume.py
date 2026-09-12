# timeout: 900
# Twelve documents, concurrently, with partial failure handled — and then checked
# against the warehouse, which is the only way to know extraction actually worked.

import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "code")
from clarity.v0_3.extract import extract      # noqa: E402

REVIEWS = sorted(Path("data/meridian/documents/quarterly-reviews").glob("*.md"))


def safely(path: Path):
    """One document's failure must not lose the other eleven."""
    try:
        return path.name, extract(path.read_text()), None
    except Exception as error:                                   # noqa: BLE001
        return path.name, None, f"{type(error).__name__}: {error}"


with ThreadPoolExecutor(max_workers=6) as pool:
    outcomes = list(pool.map(safely, REVIEWS))

ok = [(n, r) for n, r, e in outcomes if r is not None]
failed = [(n, e) for n, r, e in outcomes if r is None]
print(f"{len(ok)} extracted, {len(failed)} failed, of {len(REVIEWS)}")
for name, error in failed:
    print(f"  {name}: {error}")

# The check that matters. The documents were written from the warehouse, so every
# extracted figure must match SQL exactly. If it does not, the extraction is wrong.
con = sqlite3.connect("data/meridian/warehouse/meridian.db")
print(f"\n{'quarter':10} {'extracted':>14} {'from SQL':>14}  agrees")
mismatches = 0
for name, record in sorted(ok, key=lambda x: x[0]):
    actual = con.execute(
        "SELECT ROUND(SUM(revenue), 2) FROM v_sales WHERE year = ? AND quarter = ?",
        (record.year, record.quarter)).fetchone()[0]
    agrees = abs(record.revenue_usd - actual) < 1.0
    mismatches += not agrees
    print(f"{record.year} Q{record.quarter}   {record.revenue_usd:>14,.0f} "
          f"{actual:>14,.0f}  {'yes' if agrees else 'NO'}")
con.close()

print(f"\n{len(ok) - mismatches} of {len(ok)} agree with the warehouse to the dollar.")

losses = [(r.year, r.quarter, r.account_loss) for _, r in ok if r.account_loss]
print(f"\nQuarters reporting a named account loss: {losses}")

print()
print("Three things this listing does that a demo would not:")
print("  · one document failing does not lose the other eleven")
print("  · the work happens concurrently, because it is all waiting on the network")
print("  · the result is checked against a source of truth rather than eyeballed")
