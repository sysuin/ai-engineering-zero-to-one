# timeout: 600
# "Every chunk shares the banner, which drags them all slightly together in the embedding space."
# How far? The same passages embedded with and without repeated page furniture: Meridian's noisy
# document, and eight distinct sections of a contract given the same banner and footer.

import itertools
import re
import statistics
import sys
from pathlib import Path

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED                 # noqa: E402

client = OpenAI()
BANNER = "MERIDIAN SUPPLY CO. — CONFIDENTIAL — DO NOT DISTRIBUTE"
FOOTER = re.compile(r"^Page \d+ of \d+ \|.*$", re.M)


def embed(texts: list[str]) -> np.ndarray:
    data = client.embeddings.create(model=MODEL_EMBED, input=texts).data
    matrix = np.array([d.embedding for d in data], dtype=np.float32)
    return matrix / np.linalg.norm(matrix, axis=1, keepdims=True)


def mean_pairwise(texts: list[str]) -> float:
    vectors = embed(texts)
    return statistics.mean(float(vectors[i] @ vectors[j])
                           for i, j in itertools.combinations(range(len(texts)), 2))


def strip(text: str) -> str:
    return FOOTER.sub("", text.replace(BANNER, "")).strip()


noisy = Path("data/meridian/documents/awkward/04-header-footer-noise.md").read_text()
pages = [p.strip() for p in noisy.split(BANNER) if p.strip()]
pages = [BANNER + "\n\n" + p for p in pages]
contract = Path("data/meridian/documents/contracts/contract-MSC-2023-109.md").read_text()
sections = ["## " + s.strip() for s in re.split(r"^## ", contract, flags=re.M)[1:]]
furnished = [f"{BANNER}\n\n{s}\n\nPage {n} of {len(sections)} | Issued 2023-02-01 | Owner: Procurement"
             for n, s in enumerate(sections, 1)]

print("mean cosine similarity between every pair of passages\n")
print(f"  {'passages':<40}{'with furniture':>15}{'stripped':>10}")
for name, texts in ((f"noisy document, {len(pages)} pages", pages),
                    (f"contract sections, {len(sections)}", furnished)):
    print(f"  {name:<40}{mean_pairwise(texts):>15.3f}{mean_pairwise([strip(t) for t in texts]):>10.3f}")
banner_alone = embed([BANNER])[0]
print(f"\n  each contract section's similarity to the banner on its own:")
print(f"  {statistics.mean(float(v @ banner_alone) for v in embed(furnished)):.3f} with furniture, "
      f"{statistics.mean(float(v @ banner_alone) for v in embed([strip(t) for t in furnished])):.3f} stripped")
