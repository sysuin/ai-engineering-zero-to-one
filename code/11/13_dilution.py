# timeout: 600
# One vector per passage means a passage's vector is a blend of everything in it. The right clause
# for each of the ten questions, with growing amounts of unrelated text appended: how far does its
# similarity to the question fall, and when does a wrong clause overtake it?

import re

import numpy as np
import tiktoken
from openai import OpenAI

from _corpus import QUERIES, load
from clarity.config import MODEL_EMBED

client = OpenAI()
ENC = tiktoken.get_encoding("o200k_base")
corpus = load()
first = corpus[0]["contract"]
clauses = {c["clause"]: c["text"] for c in corpus if c["contract"] == first}

# Padding: product-specification boilerplate from the long appendix, about none of the questions.
appendix = open("data/meridian/documents/awkward/03-long-appendix.md").read()
filler = ENC.encode(" ".join(re.sub(r"#+ .*", "", appendix).split()))


def embed(texts: list[str]) -> np.ndarray:
    data = client.embeddings.create(model=MODEL_EMBED, input=texts).data
    array = np.array([d.embedding for d in data])
    return array / np.linalg.norm(array, axis=1, keepdims=True)


questions = embed([q for q, _ in QUERIES])
clean = embed(list(clauses.values()))
numbers = list(clauses)
PADDING = (0, 100, 300, 1000, 3000)

def pad(text: str, tokens: int, offset: int) -> str:
    """Append `tokens` of filler, from a different place for each clause."""
    if not tokens:
        return text
    start = offset * len(filler) // 9
    return text + "\n\n" + ENC.decode((filler * 2)[start:start + tokens])


print(f"the ten questions against the eight clauses of {first}, with unrelated")
print("specification text appended: to the right clause only, or to every clause\n")
print(f"  {'tokens':>7}{'clause':>8}{'similarity to':>15}{'right clause first':>20}{'every clause':>14}")
print(f"  {'added':>7}{'tokens':>8}{'right clause':>15}{'(only it padded)':>20}{'padded':>14}")
for tokens in PADDING:
    right = embed([pad(clauses[want], tokens, 0) for _, want in QUERIES])
    sims = np.sum(questions * right, axis=1)
    alone = sum(sims[qi] > max(clean[j] @ questions[qi]
                               for j, n in enumerate(numbers) if n != want)
                for qi, (_, want) in enumerate(QUERIES))
    everyone = embed([pad(clauses[n], tokens, j + 1) for j, n in enumerate(numbers)])
    together = sum(numbers[int(np.argmax(everyone @ questions[qi]))] == want
                   for qi, (_, want) in enumerate(QUERIES))
    size = np.mean([len(ENC.encode(pad(clauses[w], tokens, 0))) for _, w in QUERIES])
    print(f"  {tokens:>7,}{size:>8,.0f}{sims.mean():>15.3f}{alone:>17}/10{together:>11}/10")
print("\nsimilarity: mean over the ten questions, question against padded right clause")
