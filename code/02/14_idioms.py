# The idioms you will read in every listing from here on: comprehensions, sorting with a
# key, enumerate, zip, any and all, and None.

import sqlite3

con = sqlite3.connect("data/meridian/warehouse/meridian.db")
rows = con.execute("""
    SELECT region, ROUND(SUM(revenue)) FROM v_sales WHERE year = 2025 GROUP BY region
""").fetchall()
con.close()
print("rows:", rows[:2], "...")

# A list comprehension is a loop that builds a list, written as one expression.
regions = [region for region, revenue in rows]
print("\nregions:", regions)

# Add an `if` and it is a WHERE clause.
over_six_million = [region for region, revenue in rows if revenue > 6_000_000]
print("over 6m:", over_six_million)

# A dictionary comprehension builds a lookup table the same way.
revenue_by_region = {region: revenue for region, revenue in rows}
print("West:", f"{revenue_by_region['West']:,.0f}")

# sorted() takes a key: a function that says what to sort BY. This is ORDER BY 2 DESC.
ranked = sorted(revenue_by_region.items(), key=lambda pair: pair[1], reverse=True)

# enumerate() numbers things as you loop; start=1 because people count from one.
print("\nRanked")
for position, (region, revenue) in enumerate(ranked, start=1):
    print(f"  {position}. {region:10} {revenue:>12,.0f}")

# zip() walks two lists side by side, like joining on row position.
targets = [5_000_000, 8_000_000, 6_500_000, 10_000_000, 5_500_000]
print("\nAgainst target")
for (region, revenue), target in zip(rows, targets):
    print(f"  {region:10} {'met' if revenue >= target else 'MISSED':>6}")

# any() and all() ask a question of a whole list at once.
print("\nany region missed?", any(rev < tgt for (_, rev), tgt in zip(rows, targets)))
print("all above 4m?     ", all(revenue > 4_000_000 for _, revenue in rows))

# None means "no value". Test for it with `is`, never with ==.
best_margin = None
print("\nbest_margin is None:", best_margin is None)
print("revenue_by_region.get('Atlantis'):", revenue_by_region.get("Atlantis"))
