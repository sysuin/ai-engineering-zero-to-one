# timeout: 900
# Three ways to answer "the ten nearest chunks WHERE tenant = x", at filters that match from half
# of the index down to one chunk in a thousand: over-fetch and discard, filter inside the graph
# walk, or skip the graph and search the matching rows exactly.

import math
import time

import hnswlib
import numpy as np

from _synthetic import structured_data

N, DIMS, QUERIES, K, EF = 50_000, 256, 200, 10, 40
vectors, queries = structured_data(N, DIMS, QUERIES, seed=7)
rng = np.random.default_rng(7)

index = hnswlib.Index(space="ip", dim=DIMS)
index.init_index(max_elements=N, M=16, ef_construction=100, random_seed=7)
index.set_num_threads(1)
index.add_items(vectors)
print(f"{N:,} structured vectors, HNSW M=16; {QUERIES} held-out queries; ten results wanted\n")
print(f"  {'keeps':>5}  {'strategy':30}{'returned':>9}{'recall':>7}{'examined':>9}{'ms':>7}")

for share in (0.5, 0.1, 0.01, 0.001):
    allowed = rng.random(N) < share
    rows = np.flatnonzero(allowed)
    truth = [set(rows[np.argsort(-(vectors[rows] @ q))[:K]].tolist()) for q in queries]

    def report(name, results, examined, seconds=None):
        returned = np.mean([len(r) for r in results])
        recall = np.mean([len(set(np.asarray(r).tolist()) & t) / len(t)
                          for r, t in zip(results, truth)])
        ms = f"{seconds * 1000 / QUERIES:.2f}" if seconds is not None else "-"
        print(f"  {share:>5.1%}  {name:30}{returned:>9.1f}{recall:>7.0%}{examined:>9}{ms:>7}")

    # 1. Ask the graph for more than needed; discard what does not match.
    for fetch in (K, min(N, math.ceil(3 * K / share))):
        index.set_ef(max(EF, fetch))
        started = time.perf_counter()
        labels, _ = index.knn_query(queries, k=fetch)
        kept = [[i for i in row if allowed[i]][:K] for row in labels]
        report(f"post-filter, fetch {fetch:,}", kept, "-",
               time.perf_counter() - started)

    # 2. Tell the graph walk to skip rows that fail the filter.
    index.set_ef(EF)
    checked = 0

    def matches(i: int) -> bool:
        global checked
        checked += 1                     # every row the walk asks about
        return bool(allowed[i])

    labels = [index.knn_query(q[None, :], k=K, filter=matches)[0][0]
              for q in queries]
    report("filter during the walk", labels,
           f"{checked // QUERIES:,}")

    # 3. Skip the graph: score only the matching rows, exactly.
    started = time.perf_counter()
    subset = vectors[rows]
    exact = [rows[np.argsort(-(subset @ q))[:K]] for q in queries]
    report("exact search of matching rows", exact, f"{len(rows):,}",
           time.perf_counter() - started)
    print()

print("examined: rows checked against the filter (walk) or scored (exact), per query.")
print("The walk also computes a distance for every row it visits; its filter is a Python")
print("function here, so its time is not comparable and is not shown.")
