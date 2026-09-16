# What β does in direct preference optimisation, on a model small enough to see: one question,
# five candidate answers, and two thousand noisy human preferences between pairs of them.

import math

import numpy as np

rng = np.random.default_rng(30)
ANSWERS = ["terse", "correct", "cited", "chatty", "wrong"]   # "cited" = correct and cited
TRUE_QUALITY = np.array([0.0, 1.0, 2.0, 0.5, -1.0])       # what people prefer, on average
REFERENCE = np.log(np.array([0.30, 0.25, 0.10, 0.30, 0.05]))  # the model before training

# Preferences, Bradley-Terry style: a beats b with probability sigmoid(quality_a - quality_b).
pairs = []
for _ in range(2000):
    a, b = rng.choice(5, size=2, replace=False)
    if rng.random() < 1 / (1 + math.exp(TRUE_QUALITY[b] - TRUE_QUALITY[a])):
        pairs.append((a, b))
    else:
        pairs.append((b, a))
W = np.array([w for w, _ in pairs])
L = np.array([l for _, l in pairs])


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def dpo(beta: float, steps: int = 3000, lr: float = 0.5) -> np.ndarray:
    """For a categorical policy, log pi(w) - log pi(l) is theta_w - theta_l: the normaliser cancels."""
    theta = REFERENCE.copy()
    for _ in range(steps):
        margin = beta * ((theta[W] - theta[L]) - (REFERENCE[W] - REFERENCE[L]))
        weight = 1 / (1 + np.exp(margin))                  # d(-log sigmoid(m)) / dm = -(1 - sigmoid(m))
        grad = np.zeros(5)
        np.add.at(grad, W, -beta * weight)
        np.add.at(grad, L, beta * weight)
        theta -= lr * grad / len(pairs)
    return theta


def kl(p, q):
    return float(np.sum(p * np.log(p / q)))


ref = softmax(REFERENCE)
print(f"{len(pairs)} preferences; before training the model gives the best answer {ref[2]:.0%}\n")
print(f"  {'beta':>5}  " + "".join(f"{a:>9}" for a in ANSWERS) + f"{'KL from start':>15}")
print(f"  {'start':>5}  " + "".join(f"{p:>9.0%}" for p in ref) + f"{0.0:>15.2f}")
for beta in (2.0, 0.5, 0.1):
    policy = softmax(dpo(beta))
    print(f"  {beta:>5}  " + "".join(f"{p:>9.0%}" for p in policy) + f"{kl(policy, ref):>15.2f}")

closed = softmax(REFERENCE + TRUE_QUALITY / 0.5)
print(f"  {'':>5}  " + "".join(f"{p:>9.0%}" for p in closed) + f"{kl(closed, ref):>15.2f}"
      "   <- start x exp(quality / beta), beta 0.5")

print("\na small beta lets the preferences move the model far from where it started; a large one keeps it")
print("close. The same data, the same loss, and the distance travelled is a setting you choose.")
