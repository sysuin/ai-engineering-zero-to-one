# Why whole-column code is fast: the same sum, three ways, timed.

import math
import time

import numpy as np
import pandas as pd

sales = pd.read_csv("data/meridian/warehouse/transactions.csv")
qty = sales["qty"].tolist()                 # plain Python lists of Python numbers
price = sales["unit_price"].tolist()
disc = sales["discount_pct"].tolist()
q, p, d = (sales[c].to_numpy() for c in ("qty", "unit_price", "discount_pct"))


def best_ms(fn, repeat=5):
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - start)
    return result, 1000 * min(times)


def loop():
    total = 0.0
    for i in range(len(qty)):
        total += qty[i] * price[i] * (1 - disc[i] / 100)
    return total


def with_pandas():
    return (sales["qty"] * sales["unit_price"] * (1 - sales["discount_pct"] / 100)).sum()


def with_numpy():
    return float(np.sum(q * p * (1 - d / 100)))


print(f"{len(qty):,} order lines\n")
results = {}
for name, fn in [("Python loop", loop), ("pandas", with_pandas), ("NumPy", with_numpy)]:
    value, ms = best_ms(fn)
    results[name] = (value, ms)
    print(f"  {name:12} {value:>18,.6f}   {ms:8.2f} ms")

loop_ms, numpy_ms = results["Python loop"][1], results["NumPy"][1]
print(f"\nthe loop took {loop_ms / numpy_ms:,.0f} times as long as NumPy on this machine")

a, b = results["Python loop"][0], results["NumPy"][0]
print(f"loop == NumPy exactly? {a == b}   difference {abs(a - b):.2e}")
print(f"math.isclose?          {math.isclose(a, b, rel_tol=1e-12)}")
