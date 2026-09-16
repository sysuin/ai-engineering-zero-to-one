# Errors on purpose: raise them at the boundary, catch the ones you expect, and never
# swallow the ones you do not.


def parse_qty(raw: str) -> int:
    """Turn a quantity from a file into an int, or say exactly why it cannot be one."""
    try:
        qty = int(raw)
    except ValueError:
        raise ValueError(f"quantity must be a whole number, got {raw!r}") from None
    if qty <= 0:
        raise ValueError(f"quantity must be positive, got {qty}")
    return qty


for raw in ["12", " 7 ", "twelve", "-3", "4.0"]:
    try:
        print(f"{raw!r:10} -> {parse_qty(raw)}")
    except ValueError as error:
        print(f"{raw!r:10} -> rejected: {error}")

# Processing many records: keep going, but keep the failures where you can see them.
rows = [{"qty": "3", "unit_price": "10.00"}, {"qty": "x", "unit_price": "4.50"},
        {"qty": "2", "unit_price": "7.25"}]
total, rejects = 0.0, []
for number, row in enumerate(rows, start=1):
    try:
        total += parse_qty(row["qty"]) * float(row["unit_price"])
    except ValueError as error:
        rejects.append((number, str(error)))
print(f"\ntotal {total:.2f} from {len(rows) - len(rejects)} rows; rejected: {rejects}")


# The anti-pattern. `except Exception: pass` catches the error you meant to ignore AND
# the typo you did not know you had.
def total_swallowing(rows):
    total = 0.0
    for row in rows:
        try:
            total += parse_qty(row["qty"]) * float(row["unit_prise"])   # a typo
        except Exception:                                                # noqa: BLE001
            pass
    return total


print("\nswallowing total:", total_swallowing(rows), " <- every row failed, silently")

# `finally` runs whether or not an error happened — for closing, releasing, logging.
try:
    parse_qty("0")
except ValueError as error:
    print("\ncaught:", error)
finally:
    print("finally: always runs")
