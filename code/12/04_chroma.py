# timeout: 900
# The same index, in a database rather than a NumPy array.

import shutil
import sys
from pathlib import Path

import chromadb
from chromadb.config import Settings

sys.path.insert(0, "code")
from meridian_index import load_index      # noqa: E402

chunks, vectors = load_index()
STORE = Path("data/meridian/chroma")
shutil.rmtree(STORE, ignore_errors=True)

client = chromadb.PersistentClient(path=str(STORE),
                                   settings=Settings(anonymized_telemetry=False))
collection = client.create_collection(
    name="meridian",
    # Chroma will embed for you. We supply our own, so the index matches Chapter 11
    # exactly and no money is spent re-embedding.
    metadata={"hnsw:space": "cosine", "hnsw:M": 16, "hnsw:construction_ef": 100},
)

# Metadata must be flat scalars, so anything nested has to be flattened at write time.
def metadata_of(chunk: dict) -> dict:
    return {k: v for k, v in chunk.items()
            if k not in ("id", "text") and isinstance(v, (str, int, float, bool))}


BATCH = 500
for i in range(0, len(chunks), BATCH):
    window = chunks[i:i + BATCH]
    collection.add(
        ids=[c["id"] for c in window],
        documents=[c["text"] for c in window],
        embeddings=[v.tolist() for v in vectors[i:i + BATCH]],
        metadatas=[metadata_of(c) for c in window],
    )

print(f"stored {collection.count():,} chunks in {STORE}\n")

# The same pre-filtered query as the previous listing, now one call.
query_vector = vectors[[c["id"] for c in chunks].index(
    next(c["id"] for c in chunks if c.get("year") == 2024 and c.get("quarter") == 3
         and c["kind"] == "quarterly-review"))]

result = collection.query(
    query_embeddings=[query_vector.tolist()],
    n_results=3,
    where={"$and": [{"kind": "quarterly-review"}, {"year": 2024}, {"quarter": 3}]},
)
print("Filtered query — kind=quarterly-review AND year=2024 AND quarter=3")
for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0],
                               result["distances"][0]):
    print(f"  {distance:.3f}  {meta['source']:20} {meta['heading'][:36]}")

print("\nSame query, no filter")
result = collection.query(query_embeddings=[query_vector.tolist()], n_results=3)
for meta, distance in zip(result["metadatas"][0], result["distances"][0]):
    print(f"  {distance:.3f}  {meta['source']:20} {meta['heading'][:36]}")
print("  Three different quarters, all at distance zero. Every quarterly review opens")
print("  with the same title block, so those chunks are byte-identical and therefore")
print("  have identical vectors. Without the filter there is no way to prefer one.")
print("  This is why Chapter 10 insisted that a chunk carry what it is about, and it")
print("  is the near-duplicate problem Chapter 14 has to solve properly.")

# Updating a document in place, and the thing that goes wrong if you forget.
collection.upsert(ids=[chunks[0]["id"]], documents=["REVISED: " + chunks[0]["text"]],
                  embeddings=[vectors[0].tolist()],
                  metadatas=[metadata_of(chunks[0])])
stored = collection.get(ids=[chunks[0]["id"]])
print(f"\nAfter upsert, the text starts: "
      f"{stored['documents'][0][:40]!r}")
print("…and the vector is the OLD one, because we upserted text without re-embedding.")
print("Nothing errored. The document now says one thing and is findable as another.")
print()
print("Every vector store has this failure. Store the hash of the text alongside the")
print("vector, and refuse to serve a chunk whose text no longer matches it.")
