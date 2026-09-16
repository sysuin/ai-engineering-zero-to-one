# Quantisation: storing weights in fewer bits, and where the error comes from.
#
# A synthetic weight matrix shaped like a real one: mostly small values, and a handful of
# large outliers — which real models have, and which decide how well quantisation works.

import numpy as np

rng = np.random.default_rng(30)
ROWS, COLS = 1_024, 4_096
weights = rng.normal(0, 0.02, (ROWS, COLS)).astype(np.float32)
outliers = rng.random(weights.shape) < 0.001
weights[outliers] *= 25


def quantise(w: np.ndarray, bits: int, group: int | None) -> np.ndarray:
    """Symmetric round-to-nearest, with one scale per tensor or per group of weights."""
    levels = 2 ** (bits - 1) - 1
    flat = w.reshape(-1, group) if group else w.reshape(1, -1)
    scale = np.abs(flat).max(axis=1, keepdims=True) / levels
    return (np.round(flat / scale).clip(-levels, levels) * scale).reshape(w.shape)


x = rng.normal(size=(64, COLS)).astype(np.float32)          # a batch of inputs
reference = x @ weights.T

print(f"{ROWS:,} x {COLS:,} weights, {outliers.mean():.1%} of them outliers\n")
print(f"  {'format':<28} {'bits/weight':>11} {'weight error':>13} {'output error':>13}")
for name, bits, group in (("unquantised (the reference)", 16, None),
                          ("8-bit, one scale", 8, None),
                          ("4-bit, one scale", 4, None),
                          ("4-bit, a scale per 128", 4, 128),
                          ("4-bit, a scale per 32", 4, 32)):
    stored = bits + (16 / group if group else 0)              # a 16-bit scale per group
    # (the reference is float32 in memory; "16" is the width a deployed model would use)
    approx = quantise(weights, bits, group) if bits < 16 else weights
    weight_error = np.linalg.norm(approx - weights) / np.linalg.norm(weights)
    output_error = np.linalg.norm(x @ approx.T - reference) / np.linalg.norm(reference)
    print(f"  {name:<28} {stored:>11.2f} {weight_error:>13.1%} {output_error:>13.1%}")

print("\nOne scale for the whole tensor must stretch to the largest outlier, so at four")
print("bits nearly every ordinary weight rounds to zero. A scale per small group costs a")
print("fraction of a bit and keeps the outliers from setting everyone else's precision.")
print("A relative error on one matrix is not quality on a task; measure that on the eval")
print("set, because whether 12% here costs anything depends on the model and the task.")
