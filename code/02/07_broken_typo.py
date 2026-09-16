# expect-fail
# Failure 1 of 4: a name that does not exist. Usually a typo.

lines = [
    {"sku": "MRD-CLE-001", "qty": 480, "unit_price": 12.50},
    {"sku": "MRD-SAF-011", "qty": 100, "unit_price": 41.00},
]

total = 0
for line in lines:
    unit_price = line["unit_price"]
    total = total + line["qty"] * unit_prise

print(total)
