# Which of the semantic layer's metrics can be added up across which breakdowns? Not a
# property of the metric alone: it depends on the grain of the thing being counted.

import sqlite3

import yaml

LAYER = yaml.safe_load(open("code/clarity/v0_7/semantic.yaml"))
con = sqlite3.connect("file:data/meridian/warehouse/meridian.db?mode=ro", uri=True)
DIMENSIONS = ["quarter", "channel", "region", "customer", "category", "sku"]


def adds_up(sql: str, dimension: str) -> bool:
    total = con.execute(f"SELECT {sql} FROM v_sales WHERE year = 2025").fetchone()[0]
    parts = [r[0] for r in con.execute(
        f"SELECT {sql} FROM v_sales WHERE year = 2025 GROUP BY {dimension}")]
    return abs(sum(p or 0 for p in parts) - total) <= 1e-6 * max(1.0, abs(total)) + 0.01 * len(parts)


print("2025: does the sum of the parts equal the whole?\n")
print(f"  {'metric':16}" + "".join(f"{d:>10}" for d in DIMENSIONS))
matrix = {}
for name, metric in LAYER["metrics"].items():
    matrix[name] = [adds_up(metric["sql"], d) for d in DIMENSIONS]
    print(f"  {name:16}" + "".join(f"{'adds' if ok else '·':>10}" for ok in matrix[name]))

print("\nwhere a distinct count adds up, and why:")
for d, ok in zip(DIMENSIONS, matrix["orders"]):
    grain = con.execute(f"SELECT MAX(n) FROM (SELECT COUNT(DISTINCT {d}) AS n FROM v_sales "
                        f"WHERE year = 2025 GROUP BY order_id)").fetchone()[0]
    print(f"  orders by {d:9} {'adds' if ok else 'does not':9}  most {d} values in one order: {grain}")
