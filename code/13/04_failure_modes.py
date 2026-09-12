# timeout: 1200
# Four ways a RAG system gets an answer wrong, and how to tell them apart.
#
# They need different fixes, and they are indistinguishable from the answer alone.

import json
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED, MODEL_FAST     # noqa: E402
from meridian_index import load_index                  # noqa: E402

client = OpenAI()
chunks, vectors = load_index()


def embed_one(text: str) -> np.ndarray:
    v = np.array(client.embeddings.create(model=MODEL_EMBED, input=[text]).data[0]
                 .embedding, dtype=np.float32)
    return v / np.linalg.norm(v)


def retrieve(question: str, k: int = 5):
    scores = vectors @ embed_one(question)
    order = np.argsort(-scores)[:k]
    return [(chunks[i], float(scores[i])) for i in order]


def where_is_the_answer(question: str, needle: str) -> tuple[int, float]:
    """The rank of the chunk that actually contains the answer."""
    scores = vectors @ embed_one(question)
    order = np.argsort(-scores)
    for rank, i in enumerate(order, start=1):
        if needle in chunks[i]["text"]:
            return rank, float(scores[i])
    return -1, 0.0


print("Failure 1 — the right chunk was never retrieved\n")
QUESTION = "What was total revenue in 2024 Q3?"
for chunk, score in retrieve(QUESTION, 5):
    print(f"  {score:.3f}  {chunk['source']:22} {chunk['heading'][:30]}")
rank, score = where_is_the_answer(QUESTION, "Revenue for 2024 Q3 was")
print(f"\n  The chunk containing the answer is at rank {rank} ({score:.3f}).")
print("  Every quarterly review has a Summary section worded almost identically, so")
print("  the embedding recognises 'a revenue summary' perfectly and cannot tell which")
print("  quarter. Eight other quarters outrank the right one.")
print("  Fix: metadata filtering and hybrid search — Chapter 14. Not a better prompt.")

print("\n\nFailure 2 — the question had two parts and retrieval served one\n")
QUESTION = "How many days notice before a price rise, and how long to pay an invoice?"
excerpts = retrieve(QUESTION, 5)
context = "\n\n".join(f"[{c['source']}] {c['text']}" for c, _ in excerpts)
answer = (client.chat.completions.create(
    model=MODEL_FAST, temperature=0, max_completion_tokens=300,
    messages=[{"role": "user", "content":
               f"<excerpts>\n{context}\n</excerpts>\n\n{QUESTION}"}],
).choices[0].message.content or "").strip()
print(f"  retrieved: {[c['heading'][:22] for c, _ in excerpts]}")
print(f"  answer:    {answer[:200]}")
print("  All five retrieved chunks are clause 2, from five different contracts. The")
print("  payment clause was never returned, so half the question is unanswerable from")
print("  what was sent — and the model correctly says so for that half while answering")
print("  the other confidently.")
print("  One retrieval, one embedding, one ranking: a two-part question competes with")
print("  itself, and the stronger half wins.")
print("  Fix: decompose the question and retrieve per part — Chapter 9's pattern,")
print("  applied to retrieval rather than reasoning.")

print("\n\nFailure 3 — the retrieved chunk is right and says something else\n")
QUESTION = "What is the price adjustment cap?"
excerpts = retrieve(QUESTION, 3)
print(f"  retrieved from {len({c['source'] for c, _ in excerpts})} different contracts:")
for chunk, score in excerpts:
    caps = [w for w in chunk["text"].split() if w.endswith("%")]
    print(f"    {score:.3f}  {chunk.get('ref', chunk['source'])[:18]:20} caps: {caps[:3]}")
print("  The question has no single answer: forty contracts have forty caps. A system")
print("  that answers it has silently picked one.")
print("  Fix: the question needs a filter. Chapter 14 rewrites queries that lack one.")

print("\n\nFailure 4 — nothing relevant exists and it answers anyway\n")
QUESTION = "How many people work at the Columbus depot?"
excerpts = retrieve(QUESTION, 3)
context = "\n\n".join(f"[{c['source']}] {c['text']}" for c, _ in excerpts)
for style, system in (("no instruction", "Answer the question."),
                      ("told to abstain",
                       "Answer only from the excerpts. If they do not contain the "
                       "answer, reply exactly: NOT IN THE DOCUMENTS.")):
    reply = (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200,
        messages=[{"role": "system", "content": system},
                  {"role": "user", "content":
                   f"<excerpts>\n{context}\n</excerpts>\n\n{QUESTION}"}],
    ).choices[0].message.content or "").strip().replace("\n", " ")
    print(f"  {style:18} {reply[:110]}")
print("  Fix: the abstention gate from the previous listing, plus the instruction.")

print("\n\nThe diagnosis is always the same three questions, in this order:")
print("  1. Is the answer in the corpus at all?")
print("  2. If so, was its chunk retrieved?")
print("  3. If so, did the model use it?")
print()
print("Each answer points at a different fix, and you cannot tell them apart by")
print("reading the output. This is why Chapter 20 logs what was retrieved.")
