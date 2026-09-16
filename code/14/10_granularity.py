# timeout: 600
# How much text to hand back. Search small chunks, search larger windows, or search small and
# return the window around each hit — compared at the same number of results and at the same
# number of tokens, because a bigger result is easier to hit and dearer to read.

import numpy as np
import tiktoken

from _retrieval import QUESTIONS, TRUTH, chunks, embed, vectors

ENC = tiktoken.get_encoding("o200k_base")
TOKENS = [len(ENC.encode(c["text"])) for c in chunks]
Q = embed([q for q, _, _ in QUESTIONS])

# Windows: up to three consecutive chunks of the same document. Tickets are separate documents
# that share a file, so each stays on its own.
windows, current = [], []
for i, c in enumerate(chunks):
    if current and (len(current) == 3 or c["source"] != chunks[current[-1]]["source"]
                    or c["kind"] == "ticket"):
        windows.append(current)
        current = []
    current.append(i)
windows.append(current)
window_of = {i: w for w, members in enumerate(windows) for i in members}
print(f"{len(chunks)} chunks, {len(windows)} windows; mean tokens: chunk {np.mean(TOKENS):.0f}, "
      f"window {np.mean([sum(TOKENS[i] for i in w) for w in windows]):.0f}")

multi = [w for w in range(len(windows)) if len(windows[w]) > 1]
window_vectors = np.array([vectors[windows[w][0]] for w in range(len(windows))])
fresh = embed(["\n\n".join(chunks[i]["text"] for i in windows[w]) for w in multi])
window_vectors[multi] = fresh          # single-chunk windows keep the chunk's own vector


def small(q):
    return [[int(i)] for i in np.argsort(-(vectors @ q))[:50]]


def big(q):
    return [windows[int(w)] for w in np.argsort(-(window_vectors @ q))[:50]]


def small_to_big(q):
    seen, out = set(), []
    for i in np.argsort(-(vectors @ q))[:50]:
        w = window_of[int(i)]
        if w not in seen:
            seen.add(w)
            out.append(windows[w])
    return out


def neighbours(q):
    """Each hit with the chunk before and after it, from the same document."""
    out, taken = [], set()
    for i in np.argsort(-(vectors @ q))[:50]:
        i = int(i)
        if i in taken:
            continue
        group = [j for j in (i - 1, i, i + 1) if 0 <= j < len(chunks)
                 and chunks[j]["source"] == chunks[i]["source"] and chunks[i]["kind"] != "ticket"
                 and j not in taken] or [i]
        taken.update(group)
        out.append(group)
    return out


def at_k(results, truth, k):
    got = [j for r in results[:k] for j in r]
    return bool(set(got) & truth), sum(TOKENS[j] for j in got)


def at_budget(results, truth, budget):
    got, spent = [], 0
    for r in results:
        cost = sum(TOKENS[j] for j in r)
        if spent + cost > budget:
            break
        got.extend(r)
        spent += cost
    return bool(set(got) & truth)


METHODS = [("small chunks", small), ("windows of three", big),
           ("small, return window", small_to_big), ("small, add neighbours", neighbours)]
RESULTS = {name: [fn(q) for q in Q] for name, fn in METHODS}

print(f"\n  {'':24}{'recall@5':>9}{'tokens@5':>10}   recall within a budget of")
print(f"  {'':24}{'':>9}{'':>10}{'150':>8}{'300':>7}{'600':>7}")
for name, _ in METHODS:
    pairs = [at_k(r, t, 5) for r, t in zip(RESULTS[name], TRUTH)]
    recall = sum(h for h, _ in pairs) / len(pairs)
    tokens = np.mean([n for _, n in pairs])
    budgets = [sum(at_budget(r, t, b) for r, t in zip(RESULTS[name], TRUTH)) / len(TRUTH)
               for b in (150, 300, 600)]
    print(f"  {name:24}{recall:>9.0%}{tokens:>10.0f}" + "".join(f"{b:>7.0%}" for b in budgets))

print("\ntokens counted with one tokeniser; a result counts if any chunk in it answers the question")
