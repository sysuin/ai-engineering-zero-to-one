# timeout: 1200
# An attacker has Clarity's vectors — an exported index, a shared vector store — but not the text.
# With the embedding API and a list of plausible values, how much of the text comes back? Then
# the same attack on vectors made after the value was replaced with a placeholder.

import random
import re
import sqlite3
import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                   # noqa: E402
from meridian_index import load_index                    # noqa: E402

client = OpenAI()
chunks, vectors = load_index()
db = sqlite3.connect("data/meridian/warehouse/meridian.db")
rng = random.Random(29)


def embed(texts: list[str]) -> np.ndarray:
    rows = []
    for start in range(0, len(texts), 512):
        reply = client.embeddings.create(model=MODEL_EMBED, input=texts[start:start + 512])
        rows += [d.embedding for d in reply.data]
    matrix = np.array(rows, dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def attack(targets: np.ndarray, truths: list[str],
           candidates: list[str],
           probes: np.ndarray) -> tuple[int, int]:
    """Top-1 and top-10 hits, ranking candidates by similarity."""
    order = np.argsort(-(targets @ probes.T), axis=1)
    index = {c: i for i, c in enumerate(candidates)}
    ranks = [int(np.where(row == index[t])[0][0])
             for row, t in zip(order, truths)]
    return sum(r == 0 for r in ranks), sum(r < 10 for r in ranks)


def report(label: str, found: tuple[int, int], n: int, pool: int) -> None:
    print(f"  {label:<30}{found[0]:>5}/{n:<4}{found[1]:>6}/{n:<4}{min(10, pool) * n / pool:>11.1f}")


print(f"  {'what the attacker recovers':<30}{'top 1':>10}{'top 10':>11}{'by chance':>11}")

# 1. Which supplier a contract names. The warehouse lists every supplier.
suppliers = [s for (s,) in db.execute("SELECT name FROM suppliers ORDER BY name")]
named = []
for i, chunk in enumerate(chunks):
    found = [s for s in suppliers if s in chunk["text"]]
    if chunk["source"].startswith("contract") and len(found) == 1:
        named.append((i, found[0]))
rows, truths = [i for i, _ in named], [s for _, s in named]
probes = embed(suppliers)
report(f"supplier, of {len(suppliers)}", attack(vectors[rows], truths, suppliers, probes),
       len(rows), len(suppliers))
hidden = embed([chunks[i]["text"].replace(s, "[SUPPLIER]") for i, s in named])
report("  after replacing the name", attack(hidden, truths, suppliers, probes),
       len(rows), len(suppliers))

# 2. Which order a support ticket is about, among real order numbers from the warehouse.
pattern = re.compile(r"\border (\d+)\b")
tickets = [(i, pattern.search(c["text"]).group(1)) for i, c in enumerate(chunks)
           if c["source"].startswith("tickets") and pattern.search(c["text"])]
all_orders = [str(n) for (n,) in db.execute("SELECT order_id FROM orders")]
mentioned = {n for _, n in tickets}
pool = sorted(mentioned | set(rng.sample([n for n in all_orders if n not in mentioned],
                                         5_000 - len(mentioned))))
rows, truths = [i for i, _ in tickets], [n for _, n in tickets]
probes = embed([f"order {n}" for n in pool])
report(f"order number, of {len(pool):,}", attack(vectors[rows], truths, pool, probes),
       len(rows), len(pool))
hidden = embed([pattern.sub("order [ORDER]", chunks[i]["text"]) for i in rows])
report("  after replacing the number", attack(hidden, truths, pool, probes),
       len(rows), len(pool))

print("\n  top 1 / top 10 = the true value ranked first, or in the first ten;")
print("  by chance = how many a random ranking would put in the first ten")
