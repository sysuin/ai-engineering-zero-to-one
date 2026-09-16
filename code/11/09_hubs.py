# Depends on the Meridian index (meridian_index.py).
# In a high-dimensional space a few points turn up as a near neighbour of far more points than
# their share — "hubs" — and in a retrieval index a hub is a chunk that appears in answers to
# questions it has nothing to do with.

import collections
import sys

import numpy as np

sys.path.insert(0, "code")
from meridian_index import load_index                            # noqa: E402

K = 10
chunks, vectors = load_index()
similarity = vectors @ vectors.T
np.fill_diagonal(similarity, -1)                                 # a chunk is not its own neighbour
neighbours = np.argsort(-similarity, axis=1)[:, :K]
occurrences = np.bincount(neighbours.ravel(), minlength=len(chunks))

print(f"{len(chunks):,} chunks, each asked for its {K} nearest neighbours\n")
print(f"  a chunk appears on average in {occurrences.mean():.0f} other chunks' top {K}")
for q in (50, 90, 99):
    print(f"  p{q:<3} {np.percentile(occurrences, q):>5.0f}")
print(f"  most  {occurrences.max():>5}")
never = int((occurrences == 0).sum())
print(f"  never anyone's neighbour: {never} chunks ({never / len(chunks):.0%})")

top = np.argsort(-occurrences)[:5]
share = occurrences[top].sum() / occurrences.sum()
print(f"\n  the five biggest hubs take {share:.1%} of all neighbour slots, against "
      f"{5 / len(chunks):.1%} if shared evenly:")
kinds = collections.Counter(chunks[i]["kind"] for i in np.argsort(-occurrences)[:50])
for i in top:
    c = chunks[i]
    text = " ".join(c["text"].split())[:58]
    print(f"    {occurrences[i]:>4}x  {c['kind']:<18} {text}")
print(f"\n  kinds among the fifty biggest hubs: "
      + ", ".join(f"{k} {n}" for k, n in kinds.most_common()))

# How many of the big hubs are near-copies of other chunks, rather than geometry at work?
near_copy = [int((similarity[i] > 0.95).sum()) for i in np.argsort(-occurrences)[:50]]
copied = sum(1 for n in near_copy if n >= 3)
print(f"  of those fifty, {copied} have three or more near-copies (cosine above 0.95) "
      "elsewhere in the index")
print(f"\nIn a corpus this size the heaviest 'hubs' are "
      f"{'mostly' if copied > 25 else 'partly'} templated text repeated across")
print("documents — the same clause in forty contracts — which crowds every search that touches")
print("it. At millions of vectors the geometric kind appears as well: chunks that sit near the")
print("centre of the space and turn up for everything. Both are found the same way, by counting.")
