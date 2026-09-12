# A function is a name for something you do more than once.

def line_revenue(qty: int, unit_price: float, discount_pct: int = 0) -> float:
    """Revenue for one order line, after any discount."""
    return qty * unit_price * (1 - discount_pct / 100)


print(line_revenue(480, 12.50))            # no discount
print(line_revenue(480, 12.50, 10))        # ten percent off
print(line_revenue(qty=100, unit_price=41.00, discount_pct=5))


def margin_pct(revenue: float, cost: float) -> float:
    """Gross margin as a percentage. Returns 0.0 rather than dividing by zero."""
    if revenue == 0:
        return 0.0
    return 100 * (revenue - cost) / revenue


print()
print(f"{margin_pct(6000.0, 3900.0):.1f}%")
print(f"{margin_pct(0.0, 0.0):.1f}%")
