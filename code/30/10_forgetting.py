# Forgetting, on a network small enough to watch. A model is trained on a broad task, then
# fine-tuned on a narrow one. How much of the broad task does it lose, and what do the two
# standard defences — staying close to the starting weights, and mixing old examples back in —
# buy and cost?

import numpy as np

rng = np.random.default_rng(3010)
DIM, HIDDEN = 10, 32


def task(n: int, which: str) -> tuple[np.ndarray, np.ndarray]:
    """'broad': a rule over the whole input space. 'narrow': a different rule, on inputs
    from one corner of it — the house behaviour a fine-tune is for."""
    x = rng.standard_normal((n, DIM))
    if which == "narrow":
        x[:, 0] = np.abs(x[:, 0]) + 1.0
        y = (x[:, 1] - x[:, 2] > 0).astype(float)
    else:
        y = (x[:, 0] * x[:, 1] + x[:, 2] > 0).astype(float)
    return x, y


def forward(params, x):
    w1, b1, w2, b2 = params
    h = np.tanh(x @ w1 + b1)
    return h, 1 / (1 + np.exp(-(h @ w2 + b2)))


def fit(params, x, y, epochs, rate=0.3, anchor=None, strength=0.0):
    params = [p.copy() for p in params]
    for _ in range(epochs):
        w1, b1, w2, b2 = params
        h, p = forward(params, x)
        d = (p - y) / len(y)
        grads = [x.T @ ((d[:, None] * w2) * (1 - h ** 2)), ((d[:, None] * w2) * (1 - h ** 2)).sum(0),
                 h.T @ d, np.array([d.sum()])]
        for i, g in enumerate(grads):
            if anchor is not None:       # pull back towards the start
                g = g + strength * (params[i] - anchor[i])
            params[i] = params[i] - rate * g
    return params


def accuracy(params, x, y):
    return np.mean((forward(params, x)[1] > 0.5) == y)


start = [rng.standard_normal((DIM, HIDDEN)) * 0.3, np.zeros(HIDDEN),
         rng.standard_normal(HIDDEN) * 0.3, np.zeros(1)]
broad_x, broad_y = task(4_000, "broad")
narrow_x, narrow_y = task(400, "narrow")
broad_test, narrow_test = task(4_000, "broad"), task(4_000, "narrow")
base = fit(start, broad_x, broad_y, epochs=3_000)

print(f"a {DIM}-input network, trained on a broad task, then fine-tuned on a narrow one\n")
print(f"  {'':34}{'broad task':>12}{'narrow task':>13}")
print(f"  {'before fine-tuning':34}{accuracy(base, *broad_test):>12.0%}{accuracy(base, *narrow_test):>13.0%}")
old = np.random.default_rng(1).choice(len(broad_x), 400, replace=False)
mixed_x = np.vstack([narrow_x, broad_x[old]])
mixed_y = np.concatenate([narrow_y, broad_y[old]])
arms = [("fine-tuned on the narrow task", narrow_x, narrow_y, None, 0.0),
        ("  staying close to the start", narrow_x, narrow_y, base, 0.05),
        ("  staying very close", narrow_x, narrow_y, base, 0.5),
        ("  with broad examples mixed in", mixed_x, mixed_y, None, 0.0)]
for label, x, y, anchor, strength in arms:
    tuned = fit(base, x, y, epochs=1_500, anchor=anchor, strength=strength)
    print(f"  {label:34}{accuracy(tuned, *broad_test):>12.0%}{accuracy(tuned, *narrow_test):>13.0%}")
