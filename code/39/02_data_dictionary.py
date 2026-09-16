# A data dictionary for Meridian's warehouse, generated from the database itself: every column,
# its type, how many distinct values it holds, and what those values look like.

import re
import sqlite3

db = sqlite3.connect("data/meridian/warehouse/meridian.db")
TABLES = ["regions", "customers", "suppliers", "products", "orders", "order_lines"]


def describe(table: str, column: str) -> tuple[int, list[str]]:
    """How many distinct values, and either all of them or the lowest and highest."""
    distinct = db.execute(f"SELECT COUNT(DISTINCT {column}) FROM {table}").fetchone()[0]
    if distinct <= 6:
        return distinct, [str(v) for (v,) in db.execute(
            f"SELECT DISTINCT {column} FROM {table} ORDER BY {column}")]
    low, high = db.execute(f"SELECT MIN({column}), MAX({column}) FROM {table}").fetchone()
    return distinct, [f"{low} to {high}"]


for table in TABLES:
    rows = db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    print(f"{table} ({rows:,} rows)")
    for _, column, kind, _, _, key in db.execute(f"PRAGMA table_info({table})"):
        distinct, values = describe(table, column)
        line, lead = "", f"  {column + (' *' if key else ''):<15}{kind.lower():<8}{distinct:>7,}  "
        for value in values:                  # wrap long lists of values under themselves
            if line and len(line) + len(value) > 44:
                print(lead + line.rstrip(", "))
                line, lead = "", " " * len(lead)
            line += value + ", "
        print(lead + line.rstrip(", "))
    print()

print("v_sales, which Chapter 15's governed queries read, has one row per order line.")
print("Its derived columns:")
definition = db.execute("SELECT sql FROM sqlite_master WHERE name = 'v_sales'").fetchone()[0]
for line in definition.splitlines():
    if re.search(r"AS (year|quarter|revenue|cost|gross_profit),?$", line.strip()):
        print("  " + " ".join(line.split()).rstrip(","))
print("\n* primary key; distinct = how many different values the column holds")
