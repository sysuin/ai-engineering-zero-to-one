# The arithmetic of chunk size and overlap. How often does a fact of a given length get cut
# in two, and what does overlap cost to prevent it?

import math
import random

rng = random.Random(4)
DOCUMENT = 30_000                       # tokens, about the size of the appendix


def chunk_starts(size: int, overlap: int) -> list[int]:
    step = size - overlap
    return list(range(0, max(1, DOCUMENT - overlap), step))


def split_rate(size: int, overlap: int, fact: int, trials: int = 20_000) -> float:
    """Share of randomly placed facts that no single chunk contains whole."""
    starts = chunk_starts(size, overlap)
    cut = 0
    for _ in range(trials):
        begin = rng.randrange(0, DOCUMENT - fact)
        end = begin + fact
        # the chunks that could contain it start at or before `begin`
        k = min(len(starts) - 1, begin // (size - overlap))
        whole = any(starts[j] <= begin and end <= starts[j] + size
                    for j in range(max(0, k - 2), min(len(starts), k + 2)))
        cut += not whole
    return cut / trials


FACT = 40                                # a sentence, roughly
print(f"a {FACT}-token fact placed at random in a {DOCUMENT:,}-token document\n")
print(f"  {'chunk':>5} {'overlap':>7} {'chunks':>6} {'index size':>10} {'fact cut':>9} {'formula':>8}")
for size in (200, 500):
    for overlap in (0, 20, 40, 80):
        chunks = len(chunk_starts(size, overlap))
        formula = max(0.0, (FACT - overlap - 1) / (size - overlap)) if overlap < FACT else 0.0
        print(f"  {size:>5} {overlap:>7} {chunks:>6} {chunks * size / DOCUMENT:>9.2f}x "
              f"{split_rate(size, overlap, FACT):>9.1%} {formula:>8.1%}")

print("\nwith no overlap, a fact is cut roughly fact/chunk of the time; an overlap at least as")
print("long as the fact means some chunk always holds it whole — at the price of a bigger index.")
