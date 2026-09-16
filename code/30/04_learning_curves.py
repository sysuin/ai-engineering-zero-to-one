# What "too many epochs" looks like, on a model small enough to watch.
#
# Not a language model: a logistic regression over bag-of-words features, trained by
# gradient descent — the same loop, the same loss, and the same failure, at a size where
# every epoch takes a millisecond. A hundred training examples, as many features as
# examples, and only five features that actually carry the signal.

import numpy as np

rng = np.random.default_rng(30)
FEATURES, INFORMATIVE, TRAIN, HOLDOUT = 100, 5, 100, 2_000

weights_true = np.zeros(FEATURES)
weights_true[:INFORMATIVE] = 2.0


def sample(n):
    x = rng.binomial(1, 0.1, (n, FEATURES)).astype(float)
    x[:, :INFORMATIVE] = rng.binomial(1, 0.5, (n, INFORMATIVE))   # the words that matter
    p = 1 / (1 + np.exp(-(x @ weights_true - 5.0)))
    return np.hstack([x, np.ones((n, 1))]), rng.binomial(1, p)   # plus an intercept


def loss(w, x, y):
    """Cross-entropy: the average surprise at the right answer — the fine-tuning loss."""
    p = np.clip(1 / (1 + np.exp(-(x @ w))), 1e-9, 1 - 1e-9)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))


x_train, y_train = sample(TRAIN)
x_hold, y_hold = sample(HOLDOUT)
w = np.zeros(FEATURES + 1)
RATE, EPOCHS = 0.2, 800
history = []
for epoch in range(1, EPOCHS + 1):
    p = 1 / (1 + np.exp(-(x_train @ w)))
    w -= RATE * x_train.T @ (p - y_train) / TRAIN          # one full pass: one epoch
    history.append((epoch, loss(w, x_train, y_train), loss(w, x_hold, y_hold)))

best = min(history, key=lambda h: h[2])
print(f"{TRAIN} training examples, {FEATURES} features, {INFORMATIVE} of them real\n")
print(f"  {'epoch':>6} {'train loss':>11} {'holdout loss':>13}")
for epoch, train, hold in history:
    if epoch in (1, 10, 50, best[0], 200, 400, 800):
        marker = "   <- best holdout" if epoch == best[0] else ""
        print(f"  {epoch:>6} {train:>11.3f} {hold:>13.3f}{marker}")

final = history[-1]
falls = all(b[1] < a[1] for a, b in zip(history, history[1:]))
print(f"\nTraining loss fell {'every epoch' if falls else 'almost every epoch'}, to "
      f"{final[1]:.3f}. Holdout loss was lowest at epoch {best[0]}")
print(f"({best[2]:.3f}) and {final[2] / best[2]:.1f}x that by epoch {EPOCHS}. After the best "
      "epoch the model was")
print(f"learning the noise in {TRAIN} examples, and only the holdout could see it.")
