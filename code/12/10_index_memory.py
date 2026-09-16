# timeout: 900
# What an HNSW index costs in memory beyond the vectors, measured from the index hnswlib
# writes, against the rule of thumb in this chapter's sizing section.

import os
import sys
import tempfile
import time

import hnswlib
import numpy as np

sys.path.insert(0, "code/12")
from _synthetic import structured_data                           # noqa: E402

N, DIMS, K, EF = 50_000, 256, 10, 64
vectors, queries = structured_data(N, DIMS, 200, seed=12)
exact = np.argsort(-(queries @ vectors.T), axis=1)[:, :K]
raw = N * DIMS * 4

print(f"{N:,} vectors of {DIMS} dimensions: {raw / 1e6:.1f} MB as 32-bit floats\n")
results = []
print(f"  {'M':>3} {'index file':>11} {'beyond vectors':>15} {'per vector':>11} "
      f"{'rule M x 8':>11} {'recall@10':>10} {'build':>7}")
for m in (8, 16, 32):
    index = hnswlib.Index(space="ip", dim=DIMS)
    index.init_index(max_elements=N, M=m, ef_construction=100, random_seed=12)
    started = time.perf_counter()
    index.add_items(vectors, num_threads=4)
    built = time.perf_counter() - started
    index.set_ef(EF)
    labels, _ = index.knn_query(queries, k=K)
    recall = np.mean([len(set(a) & set(b)) / K for a, b in zip(labels, exact)])
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "index.bin")
        index.save_index(path)
        size = os.path.getsize(path)
    beyond = size - raw
    results.append((m, beyond, recall))
    print(f"  {m:>3} {size / 1e6:>9.1f} MB {beyond / 1e6:>12.1f} MB {beyond / N:>9.0f} B "
          f"{m * 8:>9} B {recall:>10.0%} {built:>6.1f}s")

print("\nThe file holds the vectors, the bottom layer's links, the sparse upper layers and a")
print("label per vector, so the overhead runs somewhat above the rule of thumb — and it is a")
print("fraction of the vectors themselves.")
(m1, b1, r1), (m2, b2, r2) = results[-2], results[-1]
print(f"Going from M={m1} to M={m2} multiplied the overhead by {b2 / b1:.1f} and "
      f"{'bought no recall it could show' if round(r2 * 100) <= round(r1 * 100) else f'raised recall from {r1:.0%} to {r2:.0%}'}.")
