"""Pricing helpers, and the tests in _test_pricing.py that check them."""


def line_revenue(qty: int, unit_price: float, discount_pct: int = 0) -> float:
    """Revenue for one order line, after any discount."""
    return qty * unit_price * (1 - discount_pct / 100)


def average_order_value(order_totals: list[float]) -> float:
    """The mean order value."""
    return sum(order_totals) / len(order_totals)
