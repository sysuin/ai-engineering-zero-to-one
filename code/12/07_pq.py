# timeout: 900
# Product quantisation from scratch: each vector squeezed from 1,024 bytes to 8, and what that
# costs in recall — before and after re-ranking the best candidates against the exact vectors.

import numpy as np

from _synthetic import structured_data, exact_top, recall

N, DIMS, QUERIES = 30_000, 256, 200
SUBSPACES, CODES = 8, 256                 # 8 pieces of 32 numbers, each replaced by one byte
vectors, queries = structured_data(N, DIMS, QUERIES)
truth = exact_top(vectors, queries)
rng = np.random.default_rng(3)
width = DIMS // SUBSPACES

codebooks, codes = [], np.zeros((N, SUBSPACES), dtype=np.uint8)
for s in range(SUBSPACES):
    part = vectors[:, s * width:(s + 1) * width]
    book = part[rng.choice(N, CODES, replace=False)].copy()
    for _ in range(8):                                        # k-means on this piece
        d = (part ** 2).sum(1)[:, None] - 2 * part @ book.T + (book ** 2).sum(1)[None, :]
        nearest = np.argmin(d, axis=1)
        for j in range(CODES):
            members = part[nearest == j]
            if len(members):
                book[j] = members.mean(axis=0)
    d = (part ** 2).sum(1)[:, None] - 2 * part @ book.T + (book ** 2).sum(1)[None, :]
    codes[:, s] = np.argmin(d, axis=1)
    codebooks.append(book)

reconstructed = np.hstack([codebooks[s][codes[:, s]] for s in range(SUBSPACES)])
error = np.linalg.norm(vectors - reconstructed, axis=1).mean()
print(f"float32: {DIMS * 4:,} bytes per vector; PQ code: {SUBSPACES} bytes "
      f"({DIMS * 4 // SUBSPACES}x smaller); mean reconstruction error {error:.3f}\n")

approx = queries @ reconstructed.T                      # search the compressed vectors
pq_top = [np.argpartition(-row, 10)[:10] for row in approx]
print(f"  search the codes only                recall@10 {recall(pq_top, truth):6.1%}")
for pool in (50, 200):
    reranked = []
    for q, row in zip(queries, approx):
        candidates = np.argpartition(-row, pool)[:pool]
        exact = vectors[candidates] @ q                 # re-rank against the real vectors
        reranked.append(candidates[np.argpartition(-exact, 10)[:10]])
    print(f"  codes, then re-rank the top {pool:<4}     recall@10 {recall(reranked, truth):6.1%}")
