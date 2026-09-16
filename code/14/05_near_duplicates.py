# Find near-duplicate documents before they find you. MinHash estimates how much two documents
# overlap from short signatures, so every pair can be compared without comparing full texts.

import hashlib
import re
from itertools import combinations
from pathlib import Path

docs = {p.stem.replace("contract-", ""): p.read_text()
        for p in sorted(Path("data/meridian/documents/contracts").glob("*.md"))}
SHINGLE, HASHES = 5, 128


def shingles(text: str, mask_numbers: bool) -> set[str]:
    body = text.split("## 1. Scope", 1)[-1]                  # skip the header block
    if mask_numbers:
        body = re.sub(r"\d+", "#", body)
    words = re.findall(r"[a-z0-9#]+", body.lower())
    return {" ".join(words[i:i + SHINGLE]) for i in range(len(words) - SHINGLE + 1)}


def signature(items: set[str]) -> list[int]:
    """For each of 128 seeded hashes, the smallest hash of any shingle."""
    return [min(int.from_bytes(hashlib.blake2b(f"{seed}:{s}".encode(), digest_size=8).digest(),
                               "big") for s in items) for seed in range(HASHES)]


def groups(similar: list[tuple[str, str]]) -> list[set[str]]:
    parent = {d: d for d in docs}

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x
    for a, b in similar:
        parent[root(a)] = root(b)
    out = {}
    for d in docs:
        out.setdefault(root(d), set()).add(d)
    return sorted((g for g in out.values() if len(g) > 1), key=len, reverse=True)


for label, mask in [("as written", False), ("with every number masked", True)]:
    sets = {n: shingles(t, mask) for n, t in docs.items()}
    sigs = {n: signature(s) for n, s in sets.items()}
    estimates, errors = {}, []
    for a, b in combinations(docs, 2):
        est = sum(x == y for x, y in zip(sigs[a], sigs[b])) / HASHES
        exact = len(sets[a] & sets[b]) / len(sets[a] | sets[b])
        estimates[(a, b)] = est
        errors.append(abs(est - exact))
    found = groups([pair for pair, est in estimates.items() if est >= 0.95])
    top = sorted(estimates.values(), reverse=True)
    print(f"{label}")
    print(f"  most similar pair {top[0]:.2f}; tenth most similar {top[9]:.2f}; "
          f"mean estimation error {sum(errors) / len(errors):.3f}")
    print(f"  groups at 95% or more alike: "
          + (", ".join(f"{len(g)} contracts" for g in found) if found else "none"))
    for g in found:
        print(f"    {', '.join(sorted(g))}")
    print()
