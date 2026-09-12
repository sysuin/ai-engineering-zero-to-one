# Money and floating point: the trap that makes a reconciliation report wrong by a cent.

a = 0.1 + 0.2
print("0.1 + 0.2 =", a)
print("Is that 0.3?", a == 0.3)
print("What it really is:", f"{a:.20f}")

# The fix for comparisons: ask whether the difference is small enough to ignore.
print("Close enough?", abs(a - 0.3) < 1e-9)

# The fix for money: work in whole cents, or use Decimal.
from decimal import Decimal

exact = Decimal("0.1") + Decimal("0.2")
print("\nDecimal('0.1') + Decimal('0.2') =", exact)
print("Is that 0.3?", exact == Decimal("0.3"))
