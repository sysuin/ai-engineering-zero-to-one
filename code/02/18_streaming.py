# A file too big to hold: read it a line at a time, and measure what that saves.

import csv
import tracemalloc
from pathlib import Path

CSV = Path("data/meridian/warehouse/transactions.csv")


def midwest_revenue_all_at_once() -> float:
    rows = list(csv.DictReader(CSV.read_text().splitlines()))   # every row, in memory
    return sum(int(r["qty"]) * float(r["unit_price"])
               for r in rows if r["region"] == "Midwest")


def midwest_lines(path: Path):
    """A generator: `yield` hands back one row and pauses until the next is wanted."""
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["region"] == "Midwest":
                yield row


def midwest_revenue_streaming() -> float:
    return sum(int(r["qty"]) * float(r["unit_price"]) for r in midwest_lines(CSV))


def peak_mb(fn) -> tuple[float, float]:
    tracemalloc.start()
    result = fn()
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, peak / 1e6


print(f"file on disk: {CSV.stat().st_size / 1e6:.1f} MB")
a, mem_a = peak_mb(midwest_revenue_all_at_once)
b, mem_b = peak_mb(midwest_revenue_streaming)
print(f"all at once:  revenue {a:,.2f}   peak memory {mem_a:8.2f} MB")
print(f"streaming:    revenue {b:,.2f}   peak memory {mem_b:8.2f} MB")
print(f"same answer: {abs(a - b) < 0.01}")
print(f"streaming peaked at {100 * mem_b / mem_a:.2f}% of the memory. Its peak is one row,"
      " however large the file; the other peak grows with the file.")

# A generator does nothing until something asks it for a value.
gen = midwest_lines(CSV)
print("\na generator object:", type(gen).__name__)
first = next(gen)
print("first Midwest line:", first["order_date"], first["sku"], first["qty"])
