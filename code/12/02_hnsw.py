# timeout: 900
# Approximate search: what you give up, and what you get for it.

import json
import sys
import time
from pathlib import Path

import hnswlib
import numpy as np

sys.path.insert(0, "code")
from meridian_index import load_index      # noqa: E402

chunks, real = load_index()
DIMS = real.shape[1]
rng = np.random.default_rng(11)

# 50,000 vectors, truncated to 512 numbers: large enough that the difference is visible, small enough to build
# in a chapter. The first 1,454 are Meridian; the rest are noise.
N = 50_000
DIMS_USED = 512      # Chapter 11 measured full recall at 512; building at 1536
                     # takes minutes and teaches nothing extra.
real = real[:, :DIMS_USED].copy()
real /= np.linalg.norm(real, axis=1, keepdims=True)
DIMS = DIMS_USED
extra = rng.standard_normal((N - len(real), DIMS)).astype(np.float32)
extra /= np.linalg.norm(extra, axis=1, keepdims=True)
vectors = np.vstack([real, extra])
queries = real[:50]

print(f"{N:,} vectors of {DIMS} numbers\n")

# Exact answers, to score the approximation against.
started = time.perf_counter()
exact = [set(np.argsort(-(vectors @ q))[:10].tolist()) for q in queries]
exact_ms = (time.perf_counter() - started) / len(queries) * 1000
print(f"brute force      {exact_ms:8.1f} ms per query   recall 100% by definition\n")

results = {"exact_ms": exact_ms, "build": {}, "search": {}}

for M, ef_construction in ((8, 40), (32, 200)):
    index = hnswlib.Index(space="ip", dim=DIMS)
    started = time.perf_counter()
    index.init_index(max_elements=N, M=M, ef_construction=ef_construction)
    index.add_items(vectors, np.arange(N))
    build = time.perf_counter() - started
    results["build"][f"M={M}"] = build
    print(f"M={M:<3} ef_construction={ef_construction:<4} built in {build:6.1f}s")

    for ef in (10, 25, 50, 100, 200):
        index.set_ef(ef)
        started = time.perf_counter()
        labels, _ = index.knn_query(queries, k=10)
        ms = (time.perf_counter() - started) / len(queries) * 1000
        recall = np.mean([len(set(row.tolist()) & truth) / 10
                          for row, truth in zip(labels, exact)])
        results["search"][f"M={M},ef={ef}"] = {"recall": float(recall), "ms": ms}
        speedup = exact_ms / ms
        print(f"     ef_search={ef:<4} recall@10 {recall:6.1%}   {ms:6.2f} ms   "
              f"{speedup:5.0f}x faster than exact")
    print()

Path("code/12/_hnsw.json").write_text(json.dumps(results, indent=2))

def spread(M):
    values = [v["recall"] for k, v in results["search"].items() if k.startswith(f"M={M},")]
    return max(values) - min(values)


between_m = (np.mean([v["recall"] for k, v in results["search"].items() if k.startswith("M=32,")])
             - np.mean([v["recall"] for k, v in results["search"].items() if k.startswith("M=8,")]))
print(f"Across a twentyfold range, ef_search moved recall {100 * spread(8):.1f} points at M=8 and")
print(f"{100 * spread(32):.1f} at M=32; changing M moved it {100 * between_m:.1f} points on average.")
print("Which dial binds depends on the data — and 48,546 of these vectors are random noise,")
print("the hardest data an index can be given. The next listing checks that.")
