# A list is a row. A dictionary is a record. Everything else in Python is built from these.

# A row: values in a fixed order, and you have to remember what each position means.
row = ["MRD-CLE-001", "Cleaning", 480, 12.50]

print("The SKU is", row[0])          # positions start at 0, not 1
print("The quantity is", row[2])

# A record: the same values, each with a name. You no longer have to remember.
line = {
    "sku": "MRD-CLE-001",
    "category": "Cleaning",
    "qty": 480,
    "unit_price": 12.50,
}

print("The SKU is", line["sku"])
print("The quantity is", line["qty"])

# A table is a list of records — which is exactly what a query result is.
lines = [
    {"sku": "MRD-CLE-001", "category": "Cleaning",   "qty": 480, "unit_price": 12.50},
    {"sku": "MRD-SAF-011", "category": "Safety",     "qty": 100, "unit_price": 41.00},
    {"sku": "MRD-PAC-023", "category": "Packaging",  "qty": 250, "unit_price": 3.80},
]

print()
print("The table has", len(lines), "rows")
print("The first row is", lines[0])
