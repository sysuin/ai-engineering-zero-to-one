# timeout: 900
# IVF from scratch: cluster the vectors, and at query time search only the nearest clusters.

import time

import numpy as np

from _synthetic import structured_data, exact_top, recall

N, DIMS, QUERIES, LISTS = 30_000, 256, 200, 256
vectors, queries = structured_data(N, DIMS, QUERIES)
truth = exact_top(vectors, queries)
rng = np.random.default_rng(2)

# k-means, a handful of rounds: assign every vector to its nearest centroid, move each centroid
# to the mean of its members, repeat.
centroids = vectors[rng.choice(N, LISTS, replace=False)].copy()
started = time.perf_counter()
for _ in range(8):
    assign = np.argmax(vectors @ centroids.T, axis=1)
    for j in range(LISTS):
        members = vectors[assign == j]
        if len(members):
            c = members.mean(axis=0)
            centroids[j] = c / np.linalg.norm(c)
assign = np.argmax(vectors @ centroids.T, axis=1)
lists = [np.flatnonzero(assign == j) for j in range(LISTS)]
print(f"{N:,} vectors in {LISTS} lists, trained in {time.perf_counter() - started:.1f}s; "
      f"largest list {max(map(len, lists)):,}, smallest {min(map(len, lists)):,}\n")

print(f"  {'nprobe':>6} {'scanned':>8} {'recall@10':>10} {'ms/query':>9}")
for nprobe in (1, 4, 16, 64):
    found, scanned = [], 0
    started = time.perf_counter()
    for q in queries:
        nearest_lists = np.argpartition(-(centroids @ q), nprobe)[:nprobe]
        candidates = np.concatenate([lists[j] for j in nearest_lists])
        scanned += len(candidates)
        scores = vectors[candidates] @ q
        found.append(candidates[np.argpartition(-scores, 10)[:10]])
    ms = (time.perf_counter() - started) * 1000 / QUERIES
    print(f"  {nprobe:>6} {scanned / QUERIES / N:>8.1%} {recall(found, truth):>10.1%} {ms:>9.2f}")
