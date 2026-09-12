# skip
"""
Twenty questions about Meridian's warehouse, each with a reference query.

The reference query is the ground truth. Two SQL statements that look nothing alike can
be equally correct, so the only workable way to score a generated query is to run it and
compare the result — which is what §15.11 is about.
"""

SCHEMA = """
regions(region_id, name, depot)
suppliers(supplier_id, name, country, tier)
customers(customer_id, name, region_id, segment, since)
products(sku, name, category, supplier_id, unit_cost, list_price)
orders(order_id, customer_id, order_date, channel)
order_lines(order_id, line_no, sku, qty, unit_price, unit_cost, discount_pct)

v_sales — one row per order line, already joined:
  order_id, order_date, year, quarter, channel, customer_id, customer, segment,
  region, sku, product, category, supplier, qty, unit_price, unit_cost,
  discount_pct, revenue, cost, gross_profit
"""

QUESTIONS = [
    ("What was total revenue in 2024 Q3?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE year=2024 AND quarter=3"),
    ("How many orders were placed in 2025?",
     "SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE year=2025"),
    ("Which region had the lowest revenue in 2024 Q3?",
     "SELECT region FROM v_sales WHERE year=2024 AND quarter=3 "
     "GROUP BY region ORDER BY SUM(revenue) ASC LIMIT 1"),
    ("What is our gross margin percentage for 2025?",
     "SELECT ROUND(100.0*SUM(gross_profit)/SUM(revenue),1) FROM v_sales WHERE year=2025"),
    ("How many distinct SKUs did we sell in 2023?",
     "SELECT COUNT(DISTINCT sku) FROM v_sales WHERE year=2023"),
    ("Which category made the most gross profit in 2025 Q4?",
     "SELECT category FROM v_sales WHERE year=2025 AND quarter=4 "
     "GROUP BY category ORDER BY SUM(gross_profit) DESC LIMIT 1"),
    ("What was Voss Industrial's revenue in 2025?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE supplier='Voss Industrial' "
     "AND year=2025"),
    ("How many customers are in the Midwest region?",
     "SELECT COUNT(*) FROM customers c JOIN regions r USING(region_id) "
     "WHERE r.name='Midwest'"),
    ("What was the average discount percentage in 2024?",
     "SELECT ROUND(AVG(discount_pct),3) FROM v_sales WHERE year=2024"),
    ("Which channel produced the most orders overall?",
     "SELECT channel FROM v_sales GROUP BY channel "
     "ORDER BY COUNT(DISTINCT order_id) DESC LIMIT 1"),
    ("How much revenue came from the Enterprise segment in 2024 Q2?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE segment='Enterprise' "
     "AND year=2024 AND quarter=2"),
    ("How many units of Safety products shipped in 2025 Q1?",
     "SELECT SUM(qty) FROM v_sales WHERE category='Safety' AND year=2025 AND quarter=1"),
    ("What is the total cost of goods sold across all time?",
     "SELECT ROUND(SUM(cost),2) FROM v_sales"),
    ("Which supplier has the most products?",
     "SELECT s.name FROM products p JOIN suppliers s USING(supplier_id) "
     "GROUP BY s.name ORDER BY COUNT(*) DESC LIMIT 1"),
    ("What was Southwest region revenue in 2023 Q4?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE region='Southwest' "
     "AND year=2023 AND quarter=4"),
    ("How many order lines had a discount of 15 percent?",
     "SELECT COUNT(*) FROM v_sales WHERE discount_pct=15"),
    ("What was gross profit in 2024 Q4?",
     "SELECT ROUND(SUM(gross_profit),2) FROM v_sales WHERE year=2024 AND quarter=4"),
    ("Which region grew revenue most between 2023 and 2025?",
     "SELECT region FROM (SELECT region, "
     "SUM(CASE WHEN year=2025 THEN revenue ELSE 0 END) - "
     "SUM(CASE WHEN year=2023 THEN revenue ELSE 0 END) AS growth "
     "FROM v_sales GROUP BY region) ORDER BY growth DESC LIMIT 1"),
    ("How many suppliers are based outside the United States?",
     "SELECT COUNT(*) FROM suppliers WHERE country != 'US'"),
    ("What was the average order value in 2025 Q2?",
     "SELECT ROUND(SUM(revenue)/COUNT(DISTINCT order_id),2) FROM v_sales "
     "WHERE year=2025 AND quarter=2"),
]
