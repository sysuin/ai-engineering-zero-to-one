# JSON's quiet traps: the values that do not survive, and the ones that survive wrongly.

import json
from datetime import date
from decimal import Decimal

# 1. Python will write NaN. JSON has no NaN, so most other languages cannot read it back.
text = json.dumps({"margin": float("nan")})
print("dumps(nan):          ", text)
try:
    json.dumps({"margin": float("nan")}, allow_nan=False)
except ValueError as error:
    print("allow_nan=False:     ", error)

# 2. Dates, Decimals and sets are not JSON types at all.
for value in (date(2024, 9, 30), Decimal("8461841.81"), {"Midwest", "West"}):
    try:
        json.dumps(value)
    except TypeError as error:
        print(f"dumps({type(value).__name__}):".ljust(21), error)


# ...unless you say how to turn each into something that is.
def to_jsonable(value):
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Decimal):
        return str(value)             # a string keeps every digit; a float would not
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"cannot serialise {type(value).__name__}")


record = {"as_of": date(2024, 9, 30), "revenue": Decimal("8461841.81"),
          "regions": {"West", "Midwest"}}
print("with default=:")
print("  ", json.dumps(record, default=to_jsonable))

# 3. Big integers. Python keeps every digit; a JavaScript client reads numbers as 64-bit
#    floats, which are exact only up to 2**53. An ID one past that arrives changed.
big_id = 2**53 + 1
as_float = float(json.loads(json.dumps({"id": big_id}))["id"])
print(f"\nid sent:              {big_id}")
print(f"id as a double:       {as_float:.0f}   equal? {as_float == big_id}")

# 4. Non-ASCII text is escaped by default, which is valid and hard to read.
print("\nensure_ascii default:", json.dumps({"customer": "Société Générale"}))
print("ensure_ascii=False:  ", json.dumps({"customer": "Société Générale"},
                                          ensure_ascii=False))

# 5. JSON Lines: one JSON value per line. Appendable, streamable, and how this book
#    stores tickets, traces and eval results.
lines = "\n".join(json.dumps({"n": i, "ok": i % 2 == 0}) for i in range(3))
print("\nJSON Lines:\n" + lines)
print("read back:", [json.loads(line) for line in lines.splitlines()])
