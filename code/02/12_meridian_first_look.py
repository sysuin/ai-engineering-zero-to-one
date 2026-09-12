# The question Chapter 1 promised: why did Midwest revenue fall?
# Here is half the answer — the half the warehouse knows.

import sqlite3

con = sqlite3.connect("data/meridian/warehouse/meridian.db")

rows = con.execute("""
    SELECT year, quarter, ROUND(SUM(revenue)) AS revenue
    FROM   v_sales
    WHERE  region = 'Midwest'
    GROUP  BY year, quarter
    ORDER  BY year, quarter
""").fetchall()

print("Midwest revenue by quarter")
previous = None
for year, quarter, revenue in rows:
    change = "" if previous is None else f"{100 * (revenue - previous) / previous:+7.1f}%"
    print(f"  {year} Q{quarter}   {revenue:>12,.0f}   {change}")
    previous = revenue

con.close()

print()
print("The warehouse knows the size of the fall. It does not know why.")
print("The answer is in a document, and getting to it is what Part III is about.")
