# JSON is a dictionary that travelled. Here is exactly what changes on the way.

import json

original = {
    "sku": "MRD-CLE-001",          # str   -> string
    "qty": 480,                    # int   -> number
    "unit_price": 12.5,            # float -> number
    "in_stock": True,              # bool  -> true
    "discontinued_on": None,       # None  -> null
    "tags": ["cleaning", "bulk"],  # list  -> array
    "supplier": {"name": "Voss Industrial", "tier": "Strategic"},   # dict -> object
}

as_text = json.dumps(original, indent=2)
print("As JSON text — this is what crosses the network:")
print(as_text)

back = json.loads(as_text)
print("\nAnd back again:")
for key, value in back.items():
    print(f"  {key:16} {type(value).__name__}")

print("\nIdentical after the round trip?", back == original)

# The one asymmetry worth knowing about.
keys_as_numbers = {1: "one", 2: "two"}
print("\nDictionary with integer keys:", keys_as_numbers)
print("After a JSON round trip:      ", json.loads(json.dumps(keys_as_numbers)))
print("JSON object keys are always strings. Nothing else survives.")
