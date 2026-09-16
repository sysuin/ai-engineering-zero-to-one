# Keeping an index in step with its documents without rebuilding it: key every chunk by where
# it came from, store a hash of its text, and re-embed only what changed.

import hashlib
import re
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED              # noqa: E402
from clarity.v0_4.longdoc import chunk              # noqa: E402

client = OpenAI()
calls = {"texts embedded": 0}


def embed(texts: list[str]) -> np.ndarray:
    calls["texts embedded"] += len(texts)
    data = client.embeddings.create(model=MODEL_EMBED, input=texts).data
    v = np.array([d.embedding for d in data], dtype=np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


class Index:
    def __init__(self):
        self.rows: dict[str, dict] = {}              # chunk id -> {sha, text, vector}
        self.full_rebuild_cost = 0                   # what rebuilding every time would embed

    def sync(self, documents: dict[str, str]) -> dict[str, int]:
        wanted = {}
        for source, text in documents.items():
            for n, piece in enumerate(chunk(text, max_tokens=350)):
                slug = re.sub(r"\W+", "-", piece.heading.lower())[:40]
                wanted[f"{source}#{n}:{slug}"] = piece.text
        self.full_rebuild_cost += len(wanted)
        changed = [cid for cid, text in wanted.items()
                   if cid not in self.rows or self.rows[cid]["sha"] != sha(text)]
        removed = [cid for cid in self.rows if cid not in wanted]
        if changed:
            for cid, vector in zip(changed, embed([wanted[c] for c in changed])):
                self.rows[cid] = {"sha": sha(wanted[cid]), "text": wanted[cid], "vector": vector}
        for cid in removed:
            del self.rows[cid]
        return {"kept": len(wanted) - len(changed), "embedded": len(changed), "deleted": len(removed)}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


folder = Path("data/meridian/documents/quarterly-reviews")
docs = {p.name: p.read_text() for p in sorted(folder.glob("*.md"))}

index = Index()
print("first build:        ", index.sync(docs))
print("nothing changed:    ", index.sync(docs))

edited = dict(docs)
edited["qbr-2024-Q3.md"] = edited["qbr-2024-Q3.md"].replace("Halloway Group", "Halloway Group Ltd")
print("one review edited:  ", index.sync(edited))

del edited["qbr-2023-Q1.md"]
print("one review removed: ", index.sync(edited))

print(f"\ntexts embedded in total: {calls['texts embedded']}, "
      f"against {index.full_rebuild_cost} if every sync had rebuilt the index")
stale = [cid for cid, row in index.rows.items() if sha(row["text"]) != row["sha"]]
print(f"rows whose text no longer matches its hash: {len(stale)}")
