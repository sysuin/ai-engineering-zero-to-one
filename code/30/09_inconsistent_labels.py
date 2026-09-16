# "Inconsistency is a failure mode that more data does not fix", on a model small enough to
# train in a moment. A house convention decides the ambiguous cases; one dataset is labelled by
# people who follow it, the other by five people, only one of whom does.

import numpy as np

rng = np.random.default_rng(3009)
DIM, TEST = 20, 20_000
truth = rng.standard_normal(DIM)
house = rng.standard_normal(DIM)                  # how the house settles an ambiguous case
annotators = rng.standard_normal((5, DIM))        # how five people each settle one,
annotators[0] = house                             # one of whom follows the house


def sample(n: int, consistent: bool):
    x = rng.standard_normal((n, DIM))
    score = x @ truth / np.linalg.norm(truth)
    ambiguous = np.abs(score) < 0.7            # about half of all cases
    label = score > 0
    if consistent:
        rule = house
    else:                                      # each by one of five people
        rule = annotators[rng.integers(0, 5, n)]
    rule = np.broadcast_to(rule, x.shape)
    settled = np.einsum("ij,ij->i", x, rule) > 0
    label = np.where(ambiguous, settled, label)
    return x, label.astype(float), ambiguous


def train(x: np.ndarray, y: np.ndarray, epochs: int = 300, rate: float = 0.5) -> np.ndarray:
    """Logistic regression with a small quadratic feature set, so the model can learn a
    convention that differs by region."""
    z = features(x)
    w = np.zeros(z.shape[1])
    for _ in range(epochs):
        p = 1 / (1 + np.exp(-(z @ w)))
        w -= rate * (z.T @ (p - y) / len(y) + 1e-3 * w)
    return w


def features(x: np.ndarray) -> np.ndarray:
    s = (x @ truth / np.linalg.norm(truth))[:, None]
    return np.hstack([x, x * (np.abs(s) < 0.7), np.ones((len(x), 1))])


x_test, y_test, amb_test = sample(TEST, consistent=True)   # the house convention is the truth
print("test labels follow the house convention; accuracy on the ambiguous half,")
print("and overall\n")
print(f"  {'training examples':>17}{'consistent labels':>22}{'five conventions':>22}")
for n in (50, 200, 1_000, 5_000, 20_000):
    cells = ""
    for consistent in (True, False):
        x, y, _ = sample(n, consistent)
        pred = (features(x_test) @ train(x, y)) > 0
        cells += f"{np.mean(pred[amb_test] == y_test[amb_test]):>13.0%} / {np.mean(pred == y_test):.0%}"
    print(f"  {n:>17,}{cells}")
print("\neach cell: ambiguous cases / all cases")
