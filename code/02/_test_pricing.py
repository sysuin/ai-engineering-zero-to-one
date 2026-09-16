import math

import pytest
from _pricing import average_order_value, line_revenue


def test_no_discount():
    assert line_revenue(480, 12.50) == 6000.0


def test_ten_percent_discount():
    assert math.isclose(line_revenue(480, 12.50, 10), 5400.0)


def test_average_of_three_orders():
    assert average_order_value([100.0, 200.0, 300.0]) == 200.0


def test_average_of_no_orders_is_zero():
    # A quarter with no orders is a real case, and a report should say 0, not crash.
    assert average_order_value([]) == 0.0


@pytest.mark.parametrize("qty, price, pct, expected", [
    (1, 100.0, 0, 100.0),
    (2, 50.0, 50, 50.0),
    (0, 99.0, 10, 0.0),
])
def test_line_revenue_table(qty, price, pct, expected):
    assert math.isclose(line_revenue(qty, price, pct), expected)
