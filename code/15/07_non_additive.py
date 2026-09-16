# Some metrics add up and some do not. Combining a ratio or a distinct count across groups gives
# a number that is not the total, and it is the kind of wrong a semantic layer exists to prevent.

import sqlite3

con = sqlite3.connect("file:data/meridian/warehouse/meridian.db?mode=ro", uri=True)

by_category = con.execute("""SELECT category, SUM(revenue), SUM(gross_profit), COUNT(DISTINCT order_id)
                             FROM v_sales WHERE year = 2025 GROUP BY category""").fetchall()
revenue, profit, orders, customers = con.execute(
    """SELECT SUM(revenue), SUM(gross_profit), COUNT(DISTINCT order_id), COUNT(DISTINCT customer_id)
       FROM v_sales WHERE year = 2025""").fetchone()
by_quarter = con.execute("""SELECT quarter, COUNT(DISTINCT customer_id) FROM v_sales
                            WHERE year = 2025 GROUP BY quarter""").fetchall()

print(f"2025, by category\n  {'category':11} {'revenue':>11} {'margin':>7} {'orders':>7}")
for category, r, p, o in by_category:
    print(f"  {category:11} {r:>11,.0f} {100 * p / r:>6.1f}% {o:>7,}")
print("  distinct customers by quarter: " + ", ".join(f"Q{q} {n}" for q, n in by_quarter))

avg_margin = sum(100 * p / r for _, r, p, _ in by_category) / len(by_category)
summed_orders = sum(o for *_, o in by_category)
summed_customers = sum(n for _, n in by_quarter)
print(f"\n  {'':18} {'the right total':>16} {'combining the rows':>20}")
print(f"  {'revenue':18} {revenue:>16,.0f} {sum(r for _, r, _, _ in by_category):>20,.0f}   adds up")
print(f"  {'margin':18} {100 * profit / revenue:>15.1f}% {avg_margin:>19.1f}%   average of ratios")
print(f"  {'orders':18} {orders:>16,} {summed_orders:>20,}   an order spans categories")
print(f"  {'customers':18} {customers:>16,} {summed_customers:>20,}   the same customer, every quarter")
