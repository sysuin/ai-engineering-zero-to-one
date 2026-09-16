# timeout: 900
# HyDE: search with a hypothetical answer instead of the question. Measured, on the same thirty.

from concurrent.futures import ThreadPoolExecutor

import numpy as np

from _retrieval import QUESTIONS, TRUTH, client, embed, recall_at, vectors
from clarity.config import MODEL_FAST


def hypothetical(question: str) -> str:
    return client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=120,
        messages=[{"role": "user", "content": "Write two sentences that could appear in a company "
                   f"document answering this question. Invent plausible details.\n\n{question}"}],
    ).choices[0].message.content or ""


with ThreadPoolExecutor(max_workers=10) as pool:
    fakes = list(pool.map(lambda q: hypothetical(q[0]), QUESTIONS))

Q = embed([q for q, _, _ in QUESTIONS])
H = embed(fakes)
both = Q + H
both /= np.linalg.norm(both, axis=1, keepdims=True)

TAGS = ["plain", "near-duplicate", "identifier", "paraphrase", "ticket", "obscure"]
print(f"  {'search with':22} {'all':>5} " + "".join(f"{t[:8]:>9}" for t in TAGS))
for name, M in [("the question", Q), ("a hypothetical answer", H), ("the average of both", both)]:
    orders = [list(np.argsort(-(vectors @ m))) for m in M]
    row = ""
    for tag in TAGS:
        idx = [i for i, (_, _, t) in enumerate(QUESTIONS) if t == tag]
        row += f"{sum(bool(set(orders[i][:5]) & TRUTH[i]) for i in idx) / len(idx):>9.0%}"
    print(f"  {name:22} {recall_at(orders):>5.0%} {row}")

print(f"\nexample: {QUESTIONS[18][0]!r}\n   became: {fakes[18][:120]!r}")
