# timeout: 900
# When does comparing against everything stop being acceptable?

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "code")
from meridian_index import load_index      # noqa: E402

chunks, real = load_index()
DIMS = real.shape[1]
rng = np.random.default_rng(7)

print(f"Meridian's whole corpus is {len(chunks):,} chunks of {DIMS} numbers "
      f"({real.nbytes / 1e6:.1f} MB).\n")

# Real vectors up to the size we have; synthetic beyond, because timing does not
# care what the numbers mean.
#
# Filled into one preallocated array rather than built and then stacked. An earlier
# version did `np.vstack([real, extra])`, which holds both halves and the result at
# once — three times the final size at the peak. On a machine with less headroom than
# that, the 500,000-row case swapped, and the listing reported 39 seconds: a real
# measurement of paging, and a useless one of search.
def corpus_of(n: int) -> np.ndarray:
    if n <= len(real):
        return real[:n]
    out = np.empty((n, DIMS), dtype=np.float32)
    out[:len(real)] = real
    # In blocks, so the random draw never needs a second large array either.
    for start in range(len(real), n, 50_000):
        stop = min(start + 50_000, n)
        block = rng.standard_normal((stop - start, DIMS)).astype(np.float32)
        block /= np.linalg.norm(block, axis=1, keepdims=True)
        out[start:stop] = block
    return out


query = real[0]
print(f"{'chunks':>12} {'index size':>12} {'search':>10} {'per query':>12}")
results = {}
for n in (1_000, 10_000, 100_000, 500_000):
    try:
        vectors = corpus_of(n)
    except (MemoryError, np.core._exceptions._ArrayMemoryError) as error:
        want = n * DIMS * 4 / 1e9
        print(f"{n:>12,} {want:>9.1f} GB   could not allocate — {type(error).__name__}")
        results[n] = {"bytes": int(n * DIMS * 4), "seconds": None}
        continue
    # Time several searches; one is too fast to measure reliably.
    started = time.perf_counter()
    runs = 20 if n <= 100_000 else 5
    for _ in range(runs):
        np.argsort(-(vectors @ query))[:10]
    elapsed = (time.perf_counter() - started) / runs

    results[n] = {"bytes": int(vectors.nbytes), "seconds": elapsed}
    print(f"{n:>12,} {vectors.nbytes / 1e6:>9.0f} MB {elapsed * 1000:>8.1f} ms "
          f"{elapsed * 1000:>10.1f} ms")
    del vectors

Path("code/12/_bruteforce.json").write_text(json.dumps(results, indent=2))

print()
print("Two things grow together, and the second one bites first.")
print()
print("In theory the time is linear: every query touches every vector, so ten times")
print("the corpus is ten times the work.")
print()
timed = {n: r["seconds"] for n, r in results.items() if r["seconds"]}
if len(timed) >= 2:
    (n1, t1), (n2, t2) = list(timed.items())[-2:]
    size_ratio, time_ratio = n2 / n1, t2 / t1
    gigabytes = results[n2]["bytes"] / 1e9
    print(f"In practice it is worse than linear. The last two rows are {size_ratio:.0f}x")
    print(f"more data and {time_ratio:.1f}x more time, because {gigabytes:.0f} gigabytes")
    print("stops fitting in the caches the processor is fast at reaching.")
    if time_ratio > 20 * size_ratio:
        print()
        print("That ratio is far too large to be a cache effect. This run was paging to")
        print("disk, which measures the machine rather than the algorithm — re-run it")
        print("with memory free before quoting the last row.")
print()
print("Where that happens is a property of your hardware, not of your corpus. Run this")
print("listing on your own machine and you may get different numbers — including, on a")
print("smaller one, a MemoryError rather than a slow answer. Brute force degrades")
print("gracefully right up until it stops working.")
print()
print("Brute force is exact. That is worth something, and it is why the crossover is")
print("further out than people assume: at ten thousand chunks it is instant, and a")
print("great many production systems are smaller than that.")
