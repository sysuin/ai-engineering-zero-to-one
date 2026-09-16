# The gate refused four answerable questions. Was the gate wrong, or was the answer never in
# front of it? Recompute what 03_abstention.py retrieved and look for the answer in it.

import json
import sys

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from _evalset import ANSWERABLE, UNANSWERABLE      # noqa: E402
from clarity.config import MODEL_EMBED              # noqa: E402
from meridian_index import load_index               # noqa: E402

client = OpenAI()
chunks, vectors = load_index()
RUN = json.load(open("code/13/_abstention.json"))
said_present = {r["question"]: r["grounded"] for r in RUN["scores"]}

# What a chunk must contain, or which section of which document it must be, to answer.
NEEDLES = {
    "What was total revenue in 2024 Q3?": "Revenue for 2024 Q3 was",
    "Which region was weakest in 2024 Q3?": ("qbr-2024-Q3.md", "Commentary"),
    "Which account did not renew, and when?": "Halloway Group did not renew",
    "What explanation does the company give for the Midwest fall?": "national framework agreement",
    "When did the Sanitation category launch?": "Sanitation category launched this quarter",
    "Why did gross margin fall in 2025 Q1?": "Voss Industrial applied a 12% increase",
    "What caused the spike in support contacts in 2024 Q4?": "batch defect in cloths",
    "How many days notice is required before a price rise under MSC-2022-100?":
        ("contract-MSC-2022-100.md", "2."),
    "Which law governs the supplier agreements?": "governed by the laws",
    "What happens if a supplier misses the delivery target three months running?":
        "three consecutive months entitles",
    "How long does the buyer have to reject non-conforming goods?": "reject non-conforming goods",
    "What is the racked storage capacity at Dallas?": "Racked storage",
}


def answers(chunk, needle) -> bool:
    if isinstance(needle, tuple):
        return chunk["source"] == needle[0] and chunk["heading"].startswith(needle[1])
    return needle in chunk["text"]


questions = [q for q, _ in ANSWERABLE]
Q = np.array([d.embedding for d in client.embeddings.create(
    model=MODEL_EMBED, input=questions + UNANSWERABLE).data], dtype=np.float32)
Q /= np.linalg.norm(Q, axis=1, keepdims=True)

cells = {"retrieved, gate said yes": [], "retrieved, gate said no": [],
         "not retrieved, gate said no": [], "not retrieved, gate said yes": []}
print(f"answerable questions, top {RUN['k']} excerpts")
for question, qv in zip(questions, Q):
    order = np.argsort(-(vectors @ qv))
    rank = next(r for r, i in enumerate(order, start=1) if answers(chunks[i], NEEDLES[question]))
    retrieved = rank <= RUN["k"]
    key = f"{'retrieved' if retrieved else 'not retrieved'}, gate said {'yes' if said_present[question] else 'no'}"
    cells[key].append(question)
    print(f"  rank {rank:>3}  gate {'yes' if said_present[question] else 'no ':3}  {question[:62]}")

print()
for key, items in cells.items():
    print(f"  {key:30} {len(items)}")
false_confidence = sum(said_present[q] for q in UNANSWERABLE)
print(f"  unanswerable, gate said yes    {false_confidence}")

gate_errors = len(cells["retrieved, gate said no"]) + len(cells["not retrieved, gate said yes"])
print(f"\nof {len(questions) - len(cells['retrieved, gate said yes'])} answerable questions refused or "
      f"mishandled, {len(cells['not retrieved, gate said no'])} were retrieval failures the gate "
      f"reported correctly, and {gate_errors} were the gate's own mistakes")
