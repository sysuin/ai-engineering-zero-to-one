# timeout: 600
# When an approximate index misses, which neighbours does it miss? Recall@10 counts a missed
# first-best the same as a missed tenth-best; a model reading the results does not.

import hnswlib
import numpy as np

from _synthetic import structured_data

N, DIMS, QUERIES, K = 30_000, 256, 500, 10
vectors, queries = structured_data(N, DIMS, QUERIES, seed=12)
exact = np.argsort(-(queries @ vectors.T), axis=1)[:, :K]      # true neighbours, best first

print(f"{N:,} structured vectors, {QUERIES} held-out queries; share of queries whose true")
print("r-th nearest neighbour is somewhere in the index's top 10\n")
print(f"  {'index':22}{'recall@10':>10}" + "".join(f"{'r=' + str(r):>6}" for r in (1, 2, 3, 5, 8, 10)))
for m, ef in ((4, 10), (8, 10), (16, 10), (16, 40)):
    index = hnswlib.Index(space="ip", dim=DIMS)
    index.init_index(max_elements=N, M=m, ef_construction=100, random_seed=12)
    index.set_num_threads(1)
    index.add_items(vectors)
    index.set_ef(ef)
    found, _ = index.knn_query(queries, k=K)
    # hit[q, r]: is query q's true (r+1)-th neighbour in the results?
    hit = np.array([[exact[q, r] in found[q] for r in range(K)]
                    for q in range(QUERIES)])
    cells = "".join(f"{hit[:, r - 1].mean():>6.0%}"
                    for r in (1, 2, 3, 5, 8, 10))
    print(f"  {f'M={m}, ef_search={ef}':22}{hit.mean():>10.1%}{cells}")
    if m == 4:
        weakest = found

# For the weakest index: when the true best was missed, what came back first instead?
order = np.argsort(-(queries @ vectors.T), axis=1)
rank_of = np.empty_like(order)
rank_of[np.arange(QUERIES)[:, None], order] = np.arange(N)
missed = [q for q in range(QUERIES) if exact[q, 0] not in weakest[q]]
first = np.array([rank_of[q, weakest[q][0]] + 1 for q in missed])
print(f"\nM=4, ef_search=10 missed the true best for {len(missed)} of {QUERIES} queries.")
print(f"Its first result was then at true rank {np.median(first):.0f} in the median case,")
print(f"and outside the true top 10 for {np.mean(first > K):.0%} of them.")
