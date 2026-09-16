# The same twenty generated queries, scored six ways. The metric decides the number before
# the model does.

import json
import re
import sqlite3
import sys
from decimal import Decimal

sys.path.insert(0, "code/15")
from _sqlset import QUESTIONS                # noqa: E402

DB = "data/meridian/warehouse/meridian.db"
RUN = json.load(open("code/15/_naive.json"))["rows"]
con = sqlite3.connect(DB)


def normalised(sql: str) -> str:
    return re.sub(r"\s+", " ", sql.strip().rstrip(";")).lower()


def decimals(value) -> int:
    return max(0, -Decimal(str(value)).as_tuple().exponent) if isinstance(value, float) else 0


def cell_equal(got, want, how: str) -> bool:
    if isinstance(got, (int, float)) and isinstance(want, (int, float)):
        if how == "exact":
            return got == want
        if how == "1%":
            return abs(got - want) <= 0.01 * max(1.0, abs(want))
        return round(got, decimals(want)) == want            # to the reference's precision
    return str(got).strip().lower() == str(want).strip().lower()


def rows_equal(got, want, how: str, extra_columns: bool = False) -> bool:
    if len(got) != len(want):
        return False
    for g, w in zip(got, want):
        if extra_columns:                    # every reference value appears, in order, in the row
            it = iter(g)
            if not all(any(cell_equal(x, y, how) for x in it) for y in w):
                return False
        elif len(g) != len(w) or not all(cell_equal(x, y, how) for x, y in zip(g, w)):
            return False
    return True


results = []
for (question, reference), row in zip(QUESTIONS, RUN):
    want = con.execute(reference).fetchall()
    got = con.execute(row["sql"]).fetchall()
    results.append({
        "question": question,
        "same SQL text": normalised(row["sql"]) == normalised(reference),
        "ran without error": True,
        "identical result": got == want,
        "numbers within 1%": rows_equal(got, want, "1%"),
        "numbers to the reference's precision": rows_equal(got, want, "precision"),
        "... and extra columns allowed": rows_equal(got, want, "precision", extra_columns=True),
    })

metrics = [k for k in results[0] if k != "question"]
print(f"{'scored by':40}{'right':>6}")
for m in metrics:
    print(f"  {m:38}{sum(r[m] for r in results):>6}/20")

print("\nquestions whose verdict depends on the metric:")
for r in results:
    verdicts = [r[m] for m in metrics[2:]]
    if len(set(verdicts)) > 1:
        marks = " ".join("Y" if v else "." for v in verdicts)
        print(f"  {marks}   {r['question'][:58]}")
print("  (columns: identical, within 1%, to precision, extra columns allowed)")

want = con.execute(QUESTIONS[0][1]).fetchone()[0]
print(f"\nhow loose 1% is: for '{QUESTIONS[0][0]}' ({want:,.2f}), any figure from "
      f"{want * 0.99:,.0f} to {want * 1.01:,.0f} passes")
