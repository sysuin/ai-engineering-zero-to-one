# The oldest approximate method, built in a few lines: random hyperplanes hash nearby vectors
# into the same bucket. It works, and the listing shows why graphs replaced it.

import sys
import time

import numpy as np

sys.path.insert(0, "code/12")
from _synthetic import structured_data                           # noqa: E402

N, DIMS, K = 50_000, 256, 10
vectors, queries = structured_data(N, DIMS, 200, seed=12)
exact = np.argsort(-(queries @ vectors.T), axis=1)[:, :K]
rng = np.random.default_rng(12)


def lsh(bits: int, tables: int):
    """Each table: `bits` random hyperplanes; a vector's key is which side of each it is on."""
    planes = rng.standard_normal((tables, bits, DIMS)).astype(np.float32)
    weights = 1 << np.arange(bits)
    keys = [((vectors @ p.T) > 0) @ weights for p in planes]
    buckets = []
    for key in keys:
        table: dict[int, list[int]] = {}
        for i, k in enumerate(key):
            table.setdefault(int(k), []).append(i)
        buckets.append(table)
    return planes, weights, buckets


print(f"{N:,} structured vectors of {DIMS} dimensions, {len(queries)} queries, "
      f"recall@{K} against exact search\n")
print(f"  {'bits':>4} {'tables':>6} {'recall@10':>10} {'vectors scored':>15} {'build':>7}")
rows = []
for bits, tables in ((8, 1), (8, 4), (8, 16), (12, 16), (12, 64), (16, 64)):
    started = time.perf_counter()
    planes, weights, buckets = lsh(bits, tables)
    built = time.perf_counter() - started
    found, scored = [], []
    for q, truth in zip(queries, exact):
        candidates = set()
        for p, table in zip(planes, buckets):
            candidates.update(table.get(int(((p @ q) > 0) @ weights), []))
        ids = np.fromiter(candidates, dtype=np.int64)
        best = ids[np.argsort(-(vectors[ids] @ q))[:K]] if len(ids) else ids
        found.append(len(set(best) & set(truth)) / K)
        scored.append(len(ids) / N)
    rows.append((bits, tables, float(np.mean(found)), float(np.mean(scored))))
    print(f"  {bits:>4} {tables:>6} {np.mean(found):>10.0%} {np.mean(scored):>14.1%} "
          f"{built:>6.1f}s")

best = max(rows, key=lambda r: r[2])
memory = N * best[1] * 16 / 1e6          # a bucket id and a vector id per vector, per table
print(f"\nThe best row here found {best[2]:.0%} of the true neighbours while scoring "
      f"{best[3]:.1%} of the index, using")
print(f"{best[1]} tables — about {memory:.0f} MB of hash tables for {N:,} vectors, before the "
      "vectors themselves.")
print("More bits make buckets smaller, so fewer vectors are scored and true neighbours are split")
print("across buckets; more tables win them back with several independent hashes, and every")
print("table is another copy of the index's bookkeeping. Graph indexes reach higher recall for")
print("less memory on data like this, which is why they replaced hashing in vector databases.")
