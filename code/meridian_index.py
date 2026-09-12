"""
The Meridian corpus, chunked and embedded once, cached on disk.

Chapters 12, 13 and 14 all need the same index. Embedding it on every run would be slow
and would spend money re-learning something that has not changed, so it is built once
into `data/meridian/index/` and loaded from there afterwards.

    from meridian_index import load_index
    chunks, vectors = load_index()
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from clarity.config import MODEL_EMBED          # noqa: E402
from clarity.v0_4.longdoc import chunk          # noqa: E402

DOCS = Path("data/meridian/documents")
CACHE = Path("data/meridian/index")
SOURCES = [
    ("quarterly-review", DOCS / "quarterly-reviews"),
    ("contract", DOCS / "contracts"),
    ("awkward", DOCS / "awkward"),
]


def build_chunks() -> list[dict]:
    """Every Meridian document, chunked on structure, with metadata attached."""
    out: list[dict] = []
    for kind, folder in SOURCES:
        for path in sorted(folder.glob("*.md")):
            for piece in chunk(path.read_text(), max_tokens=350):
                meta = {"kind": kind, "source": path.name,
                        "heading": piece.heading[:60]}
                if kind == "quarterly-review":
                    year, quarter = path.stem.split("-")[1], path.stem.split("-")[2]
                    meta |= {"year": int(year), "quarter": int(quarter[1:])}
                if kind == "contract":
                    meta |= {"ref": path.stem.replace("contract-", "")}
                out.append({"id": f"{path.stem}#{len(out)}", "text": piece.text, **meta})

    # Support tickets are short; each is its own chunk.
    tickets = DOCS / "tickets" / "tickets.jsonl"
    for line in tickets.read_text().splitlines():
        row = json.loads(line)
        out.append({"id": row["ticket_id"], "text": row["body"], "kind": "ticket",
                    "source": "tickets.jsonl", "heading": row["category"],
                    "category": row["category"], "year": row["year"],
                    "quarter": row["quarter"], "sku": row["sku"]})
    return out


def load_index(rebuild: bool = False) -> tuple[list[dict], np.ndarray]:
    """Chunks and their unit-normalised embeddings, built on first use."""
    meta_path, vec_path = CACHE / "chunks.json", CACHE / "vectors.npy"
    if not rebuild and meta_path.exists() and vec_path.exists():
        stored = json.loads(meta_path.read_text())
        if stored["model"] == MODEL_EMBED:
            return stored["chunks"], np.load(vec_path)

    from openai import OpenAI
    client = OpenAI()
    chunks = build_chunks()
    print(f"embedding {len(chunks)} chunks with {MODEL_EMBED} …", file=sys.stderr)

    vectors = []
    texts = [c["text"] for c in chunks]
    for i in range(0, len(texts), 256):
        response = client.embeddings.create(model=MODEL_EMBED, input=texts[i:i + 256])
        vectors.extend(item.embedding for item in response.data)

    array = np.array(vectors, dtype=np.float32)
    array /= np.linalg.norm(array, axis=1, keepdims=True)

    CACHE.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps({"model": MODEL_EMBED, "chunks": chunks}))
    np.save(vec_path, array)
    return chunks, array


if __name__ == "__main__":
    # `python3 code/meridian_index.py --build` — the one command that costs money, made
    # explicit so it is never run by an import.
    import argparse

    parser = argparse.ArgumentParser(description="Build Meridian's vector index.")
    parser.add_argument("--build", action="store_true",
                        help="rebuild even if a cached index exists")
    options = parser.parse_args()
    chunks, vectors = load_index(rebuild=options.build)
    print(f"{len(chunks):,} chunks, {vectors.shape[1]} dimensions, "
          f"{vectors.nbytes / 1e6:.1f} MB")
