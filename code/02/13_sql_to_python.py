# The same question, four ways: SQL, plain Python, pandas, and back to SQL from Python.
# They agree, because they are asking the same thing of the same data.

import csv
import sqlite3

import pandas as pd

QUARTER = (2025, 4)

# 1. Plain Python over the CSV — loops and a dictionary of totals.
totals = {}
with open("data/meridian/warehouse/transactions.csv", newline="") as f:
    for row in csv.DictReader(f):
        if (int(row["year"]), int(row["quarter"])) != QUARTER:
            continue
        revenue = int(row["qty"]) * float(row["unit_price"])
        totals[row["region"]] = totals.get(row["region"], 0.0) + revenue

plain = sorted(totals.items(), key=lambda kv: -kv[1])
print("Plain Python")
for region, revenue in plain:
    print(f"  {region:11} {revenue:>14,.2f}")

# 2. pandas — the same thing, said once.
sales = pd.read_csv("data/meridian/warehouse/transactions.csv")
sales["revenue"] = sales["qty"] * sales["unit_price"]
frame = (sales[(sales.year == QUARTER[0]) & (sales.quarter == QUARTER[1])]
         .groupby("region")["revenue"].sum().sort_values(ascending=False))
print("\npandas")
for region, revenue in frame.items():
    print(f"  {region:11} {revenue:>14,.2f}")

# 3. SQL, run from Python — the database does the work.
con = sqlite3.connect("data/meridian/warehouse/meridian.db")
rows = con.execute("""
    SELECT region, SUM(revenue) AS revenue
    FROM   v_sales
    WHERE  year = ? AND quarter = ?
    GROUP  BY region
    ORDER  BY revenue DESC
""", QUARTER).fetchall()
con.close()
print("\nSQL")
for region, revenue in rows:
    print(f"  {region:11} {revenue:>14,.2f}")

# Do they agree?
agree = all(abs(a[1] - b[1]) < 0.01 and a[0] == b[0] for a, b in zip(plain, rows))
print(f"\nAll three agree: {agree}")
