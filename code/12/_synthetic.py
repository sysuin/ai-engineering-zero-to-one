# skip
"""Two datasets for Chapter 12's benchmarks: random directions, and structured ones more like text."""
from __future__ import annotations

import numpy as np


def unit(x: np.ndarray) -> np.ndarray:
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def random_data(n: int, dims: int, queries: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    return unit(rng.standard_normal((n, dims))), unit(rng.standard_normal((queries, dims)))


def structured_data(n: int, dims: int, queries: int, intrinsic: int = 16, noise: float = 0.1,
                    seed: int = 0):
    """
    Points that vary along only a few underlying directions, with a little noise on top.

    Text embeddings behave like this: they have hundreds or thousands of coordinates, but the
    ways real passages differ from one another occupy far fewer dimensions. Queries are drawn
    the same way and are not in the index.
    """
    rng = np.random.default_rng(seed)
    basis = rng.standard_normal((dims, intrinsic))

    def draw(k):
        core = rng.standard_normal((k, intrinsic)) @ basis.T
        return unit(core + noise * np.sqrt(intrinsic / dims) * rng.standard_normal((k, dims)))
    return draw(n), draw(queries)


def exact_top(vectors: np.ndarray, queries: np.ndarray, k: int = 10) -> list[set[int]]:
    scores = queries @ vectors.T
    return [set(np.argpartition(-row, k)[:k].tolist()) for row in scores]


def recall(found, truth) -> float:
    return float(np.mean([len(set(f) & t) / len(t) for f, t in zip(found, truth)]))
