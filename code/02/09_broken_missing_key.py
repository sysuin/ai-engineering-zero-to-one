# expect-fail
# Failure 3 of 4: a key that is not in the record. Real data is not uniform.

lines = [
    {"sku": "MRD-CLE-001", "qty": 480, "unit_price": 12.50, "discount_pct": 10},
    {"sku": "MRD-SAF-011", "qty": 100, "unit_price": 41.00},   # no discount recorded
]

for line in lines:
    net = line["qty"] * line["unit_price"] * (1 - line["discount_pct"] / 100)
    print(f"{line['sku']}  {net:,.2f}")
