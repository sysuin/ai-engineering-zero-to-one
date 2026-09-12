# pandas in five minutes, for someone who already thinks in tables.
# A DataFrame is a list of records that knows it is a table.

import pandas as pd

pd.set_option("display.width", 88)
pd.set_option("display.max_columns", 12)

sales = pd.read_csv("data/meridian/warehouse/transactions.csv")

print("Shape (rows, columns):", sales.shape)
print("\nColumns:", list(sales.columns))

print("\nFirst three rows:")
print(sales[["order_date", "region", "category", "qty", "unit_price"]].head(3))

# A calculated column — the whole column at once, no loop.
sales["revenue"] = sales["qty"] * sales["unit_price"]

# WHERE region = 'Midwest'
midwest = sales[sales["region"] == "Midwest"]
print(f"\nMidwest rows: {len(midwest):,} of {len(sales):,}")

# SELECT category, SUM(revenue) ... GROUP BY category ORDER BY 2 DESC
by_category = (sales.groupby("category")["revenue"]
                    .sum()
                    .sort_values(ascending=False))
print("\nRevenue by category")
print(by_category.map(lambda v: f"{v:,.0f}").to_string())
