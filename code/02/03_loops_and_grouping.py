# A loop is what you do instead of dragging a formula down a column.
# A dictionary of totals is what you do instead of GROUP BY.

lines = [
    {"sku": "MRD-CLE-001", "category": "Cleaning",  "qty": 480, "unit_price": 12.50},
    {"sku": "MRD-SAF-011", "category": "Safety",    "qty": 100, "unit_price": 41.00},
    {"sku": "MRD-PAC-023", "category": "Packaging", "qty": 250, "unit_price":  3.80},
    {"sku": "MRD-CLE-004", "category": "Cleaning",  "qty": 240, "unit_price":  9.25},
]

# Dragging the formula down: one new value per row.
print("Revenue per line")
for line in lines:
    revenue = line["qty"] * line["unit_price"]
    print(f"  {line['sku']:14} {revenue:9,.2f}")

# SELECT SUM(qty * unit_price) FROM lines
total = 0
for line in lines:
    total = total + line["qty"] * line["unit_price"]
print(f"\nTotal revenue {total:,.2f}")

# SELECT category, SUM(qty * unit_price) FROM lines GROUP BY category
by_category = {}
for line in lines:
    revenue = line["qty"] * line["unit_price"]
    category = line["category"]
    by_category[category] = by_category.get(category, 0) + revenue

print("\nRevenue by category")
for category, revenue in by_category.items():
    print(f"  {category:12} {revenue:9,.2f}")
