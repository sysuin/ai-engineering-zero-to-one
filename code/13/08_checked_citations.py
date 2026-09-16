# timeout: 900
# Citations you can check: every claim carries a quote and the source it came from, and code
# keeps only the claims whose quote is really in that source's retrieved text.

import re
import sys
from concurrent.futures import ThreadPoolExecutor

from pydantic import BaseModel, Field

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST       # noqa: E402
from clarity.v0_5.rag import Clarity         # noqa: E402
from meridian_index import load_index        # noqa: E402

chunks, vectors = load_index()
clarity = Clarity(chunks, vectors)


class Claim(BaseModel):
    statement: str
    source: str = Field(description="The bracketed source label of the excerpt it came from.")
    quote: str = Field(description="Words copied exactly from that excerpt that support it.")


class CitedAnswer(BaseModel):
    claims: list[Claim]


QUESTIONS = [
    ("Which account did not renew, and what caused it?", None),
    ("What was total revenue in 2024 Q3, and which region was weakest?",
     {"kind": "quarterly-review", "year": 2024, "quarter": 3}),
    ("What is the utilisation of each depot at year end?", None),
    ("Which line of the Columbus goods received note was short, and by how much?", None),
    ("What approval is needed for large purchase orders?", None),
    ("What happens if a supplier misses its delivery target?", None),
]


def norm(s: str) -> str:
    return re.sub(r"[\s*_|]+", " ", s).strip().strip('"“”').lower()


def answer(question, where):
    passages = clarity.retrieve(question, k=6, where=where)
    context = clarity._context(passages)
    parsed = clarity.client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=800, response_format=CitedAnswer,
        messages=[{"role": "system", "content": "Answer only from the excerpts. Every claim must "
                   "cite its excerpt's source label and quote it exactly."},
                  {"role": "user", "content": f"<excerpts>\n{context}\n</excerpts>\n\n"
                   f"Question: {question}"}],
    ).choices[0].message.parsed
    by_source = {}
    for p in passages:
        by_source.setdefault(p.source, []).append(norm(p.text))
    kept, dropped = [], []
    for claim in parsed.claims:
        label = re.search(r"[\w.-]+\.(?:md|jsonl)", claim.source)
        texts = by_source.get(label.group(0), []) if label else []
        (kept if any(norm(claim.quote) in t for t in texts) else dropped).append(claim)
    return kept, dropped


with ThreadPoolExecutor(max_workers=6) as pool:
    results = list(pool.map(lambda q: answer(*q), QUESTIONS))

total_kept = total_dropped = 0
for (question, _), (kept, dropped) in zip(QUESTIONS, results):
    total_kept += len(kept)
    total_dropped += len(dropped)
    print(f"{question}\n   {len(kept)} claim(s) verified, {len(dropped)} dropped")
    for claim in dropped:
        print(f"     dropped: {claim.statement[:60]!r}  quote {claim.quote[:40]!r} "
              f"from {claim.source!r}")
print(f"\n{total_kept} verified, {total_dropped} dropped, across {len(QUESTIONS)} questions")
