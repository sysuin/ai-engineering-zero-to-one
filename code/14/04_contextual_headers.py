# timeout: 900
# Twelve quarterly summaries worded alike embed alike. Put what each chunk is ABOUT into the
# text that gets embedded — document, period, section — and see what dense search does.

import json
from pathlib import Path

import numpy as np

from _retrieval import QUESTIONS, TRUTH, bm25_order, chunks, embed, recall_at, rrf, vectors


def header(c: dict) -> str:
    parts = [c["source"].removesuffix(".md")]
    if c.get("year"):
        parts.append(f"{c['year']} Q{c['quarter']}")
    if c.get("ref"):
        parts.append(f"agreement {c['ref']}")
    parts.append(c.get("heading", ""))
    return " · ".join(p for p in parts if p)


print("embedding", len(chunks), "chunks again, each with a one-line header ...")
with_headers = embed([f"{header(c)}\n\n{c['text']}" for c in chunks])
Q = embed([q for q, _, _ in QUESTIONS])

plain = [[int(i) for i in np.argsort(-(vectors @ q))] for q in Q]
headed = [[int(i) for i in np.argsort(-(with_headers @ q))] for q in Q]
keyword = [bm25_order(question) for question, _, _ in QUESTIONS]

print(f"\nexample header: {header(chunks[next(i for i, c in enumerate(chunks) if c.get('year') == 2024 and c.get('quarter') == 3)])!r}\n")
print(f"  {'':22} {'all 30':>7} " + "".join(f"{t:>16}" for t in ("near-duplicate", "identifier", "paraphrase")))
saved = {}
for name, orders in [("dense, text only", plain), ("dense, with headers", headed),
                     ("hybrid, text only", [rrf(d[:50], k[:50], k=1) for d, k in zip(plain, keyword)]),
                     ("hybrid, with headers", [rrf(d[:50], k[:50], k=1) for d, k in zip(headed, keyword)])]:
    by_tag = {}
    for tag in ("near-duplicate", "identifier", "paraphrase"):
        idx = [i for i, (_, _, t) in enumerate(QUESTIONS) if t == tag]
        by_tag[tag] = sum(bool(set(orders[i][:5]) & TRUTH[i]) for i in idx) / len(idx)
    print(f"  {name:22} {recall_at(orders):>7.0%} " + "".join(f"{v:>16.0%}" for v in by_tag.values()))
    saved[name] = recall_at(orders)

Path("code/14/_headers.json").write_text(json.dumps(saved, indent=2))
