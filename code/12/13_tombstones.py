# timeout: 900
# Deleting from an HNSW index by marking rows deleted: the rows stay in the graph and are skipped
# in results. What that does to recall and speed as the dead share grows, and what a rebuild buys.

import time

import hnswlib
import numpy as np

from _synthetic import structured_data

N, DIMS, QUERIES, K, EF = 40_000, 256, 300, 10, 20
vectors, queries = structured_data(N, DIMS, QUERIES, seed=13)
order = np.random.default_rng(13).permutation(N)          # the order rows are deleted in


def build(ids: np.ndarray) -> hnswlib.Index:
    index = hnswlib.Index(space="ip", dim=DIMS)
    index.init_index(max_elements=N, M=16, ef_construction=100, random_seed=13)
    index.set_num_threads(1)
    index.add_items(vectors[ids], ids)
    index.set_ef(EF)
    return index


def measure(index: hnswlib.Index, live: np.ndarray) -> tuple[float, float]:
    truth = [set(live[np.argsort(-(vectors[live] @ q))[:K]].tolist()) for q in queries]
    started = time.perf_counter()
    labels, _ = index.knn_query(queries, k=K)
    ms = (time.perf_counter() - started) * 1000 / QUERIES
    return float(np.mean([len(set(r) & t) / K for r, t in zip(labels, truth)])), ms


index = build(np.arange(N))
print(f"{N:,} structured vectors, HNSW M=16; recall@10 against exact search over")
print(f"the live rows. Marked: deleted rows stay in the graph, ef_search={EF}. Rebuilt:")
print("a new index of the live rows, at the smallest ef_search matching that recall\n")
print(f"  {'':7}{'marked deleted':>27}   {'rebuilt':>30}")
print(f"  {'deleted':>7}{'rows held':>11}{'recall':>9}{'ms':>7}   {'rows held':>9}{'ef':>5}{'recall':>9}{'ms':>7}")
done = 0
for share in (0.0, 0.25, 0.5, 0.75, 0.9):
    target = int(share * N)
    for i in order[done:target]:
        index.mark_deleted(int(i))
    done = target
    live = np.sort(order[target:])
    recall, ms = measure(index, live)
    fresh = build(live)
    for ef in (20, 30, 40, 60, 80, 120):
        fresh.set_ef(ef)
        fresh_recall, fresh_ms = measure(fresh, live)
        if fresh_recall >= recall:
            break
    print(f"  {share:>7.0%}{N:>11,}{recall:>9.1%}{ms:>7.3f}   "
          f"{len(live):>9,}{ef:>5}{fresh_recall:>9.1%}{fresh_ms:>7.3f}")
