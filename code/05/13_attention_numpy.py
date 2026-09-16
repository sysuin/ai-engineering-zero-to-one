# Scaled dot-product attention, in eight lines of NumPy — and two properties that shape
# how every model is built.

import numpy as np

rng = np.random.default_rng(0)
tokens, d = 5, 8
X = rng.normal(size=(tokens, d))                         # one row per token
Wq, Wk, Wv = (rng.normal(size=(d, d)) for _ in range(3))  # learned in a real model


def attention(X, causal=False):
    Q, K, V = X @ Wq, X @ Wk, X @ Wv
    scores = Q @ K.T / np.sqrt(d)                        # every token against every token
    if causal:
        scores = np.where(np.triu(np.ones_like(scores), k=1) == 1, -np.inf, scores)
    weights = np.exp(scores - scores.max(axis=1, keepdims=True))
    weights /= weights.sum(axis=1, keepdims=True)        # softmax, row by row
    return weights @ V, weights


out, weights = attention(X)
np.set_printoptions(precision=2, suppress=True)
print("attention weights (row = token looking, column = token looked at):")
print(weights)
print("each row sums to", weights.sum(axis=1))

# 1. Attention alone does not know word order. Shuffle the tokens, and the outputs are
#    the same vectors, shuffled the same way.
order = np.array([3, 0, 4, 1, 2])
shuffled_out, _ = attention(X[order])
print("\nshuffled input gives shuffled output, unchanged otherwise:",
      np.allclose(shuffled_out, out[order]))

# 2. A causal mask stops each token looking ahead — which is what lets a model be trained
#    to predict the next token without seeing it.
_, causal = attention(X, causal=True)
print("\ncausal weights (nothing above the diagonal):")
print(causal)

# 3. The cost: a tokens x tokens matrix. Double the sequence, four times the entries.
for n in (1_000, 2_000, 4_000):
    print(f"  {n:>5} tokens -> {n * n:>12,} attention scores per head, per layer")
