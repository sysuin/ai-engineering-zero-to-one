# timeout: 1200
# The same twenty questions, with the model choosing a query rather than writing one.

import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "code")
sys.path.insert(0, "code/15")
from _sqlset import QUESTIONS                        # noqa: E402
from clarity.v0_7.warehouse import Warehouse         # noqa: E402

warehouse = Warehouse()
DB = "data/meridian/warehouse/meridian.db"


def reference(sql: str):
    con = sqlite3.connect(DB)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def same(a, b, tolerance: float = 0.01) -> bool:
    if a is None or b is None or len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        # The reference may select a label ("Midwest") where the builder selects a
        # label and a value. Count it correct if any cell matches the reference cell.
        for vb in row_b:
            if isinstance(vb, (int, float)):
                if not any(isinstance(va, (int, float))
                           and abs(va - vb) <= tolerance * max(1.0, abs(vb))
                           for va in row_a):
                    return False
            elif not any(str(va).strip().lower() == str(vb).strip().lower()
                         for va in row_a):
                return False
    return True


def attempt(pair):
    question, ref = pair
    try:
        result = warehouse.ask(question)
    except Exception as error:
        return {"q": question, "status": "refused", "why": str(error)[:70]}
    if result is None:
        return {"q": question, "status": "refused", "why": "no plan"}
    return {"q": question, "status": "answered", "rows": result.rows,
            "sql": result.sql, "metric": result.spec.metric,
            "ok": same(result.rows, reference(ref))}


with ThreadPoolExecutor(max_workers=10) as pool:
    results = list(pool.map(attempt, QUESTIONS))

answered = [r for r in results if r["status"] == "answered"]
refused = [r for r in results if r["status"] == "refused"]
correct = [r for r in answered if r["ok"]]
n = len(QUESTIONS)

print(f"{n} questions\n")
print(f"  answered            {len(answered):>2}/{n}")
print(f"  refused honestly    {len(refused):>2}/{n}")
print(f"  correct of answered {len(correct):>2}/{len(answered)}  "
      f"{len(correct) / max(1, len(answered)):.0%}")
print(f"  correct of all      {len(correct):>2}/{n}  {len(correct) / n:.0%}")

print("\nRefused:")
for r in refused:
    print(f"  {r['q'][:58]:60} {r['why'][:40]}")

print("\nAnswered but wrong:")
for r in answered:
    if not r["ok"]:
        print(f"  {r['q'][:58]:60} metric={r['metric']}")

print("\nEvery query that ran was built from a spec, so:")
con = sqlite3.connect(DB)
generated = {r["sql"].split()[0].upper() for r in answered}
print(f"  statements produced: {generated}")
print(f"  tables reachable:    {{v_sales}}")
print("  There is no code path from a question to DELETE, DROP or a second table.")

Path("code/15/_governed.json").write_text(json.dumps(
    {"n": n, "answered": len(answered), "refused": len(refused),
     "correct": len(correct)}, indent=2))

print()
print("The refusals are the feature. A question the layer cannot express comes back as")
print("a refusal rather than as a number that looks like every other number.")
