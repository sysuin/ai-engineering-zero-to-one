# timeout: 900
# The same index, the same settings, two kinds of data. Random directions are the hardest thing
# an approximate index can be given; text embeddings have structure, and behave very differently.

import time

import hnswlib
import numpy as np

from _synthetic import structured_data, exact_top, random_data, recall

N, DIMS, QUERIES = 30_000, 256, 200

print(f"{N:,} vectors of {DIMS} numbers; {QUERIES} queries that are NOT in the index\n")
print(f"  {'data':10} {'ef_search':>9} {'recall@10':>10} {'ms/query':>9}")
for name, make in [("random", random_data), ("structured", structured_data)]:
    vectors, queries = make(N, DIMS, QUERIES)
    truth = exact_top(vectors, queries)
    index = hnswlib.Index(space="ip", dim=DIMS)
    index.init_index(max_elements=N, M=16, ef_construction=100)
    index.set_num_threads(1)
    index.add_items(vectors)
    for ef in (10, 40, 160):
        index.set_ef(ef)
        started = time.perf_counter()
        labels, _ = index.knn_query(queries, k=10)
        ms = (time.perf_counter() - started) * 1000 / QUERIES
        print(f"  {name:10} {ef:>9} {recall(labels, truth):>10.1%} {ms:>9.2f}")
    print()
