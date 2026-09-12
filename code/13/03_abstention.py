# timeout: 1200
# The question that decides whether a RAG system can be trusted:
# how does it know when the answer is not there?

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, Field

sys.path.insert(0, "code")
from _evalset import ANSWERABLE, UNANSWERABLE      # noqa: E402
from clarity.config import MODEL_EMBED, MODEL_FAST  # noqa: E402
from meridian_index import load_index               # noqa: E402

client = OpenAI()
chunks, vectors = load_index()
QUESTIONS = [(q, True) for q, _ in ANSWERABLE] + [(q, False) for q in UNANSWERABLE]
K = 5


def embed(texts: list[str]) -> np.ndarray:
    array = np.array([d.embedding for d in client.embeddings.create(
        model=MODEL_EMBED, input=texts).data], dtype=np.float32)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


query_vectors = embed([q for q, _ in QUESTIONS])
retrieved = [[chunks[i] for i in np.argsort(-(vectors @ qv))[:K]]
             for qv in query_vectors]
top_scores = [float(np.max(vectors @ qv)) for qv in query_vectors]


# ---------------------------------------------------------------- signal 1: similarity
print("Signal 1 — the similarity of the best chunk\n")
yes = [s for s, (_, a) in zip(top_scores, QUESTIONS) if a]
no = [s for s, (_, a) in zip(top_scores, QUESTIONS) if not a]
print(f"  answerable    min {min(yes):.3f}   mean {np.mean(yes):.3f}   max {max(yes):.3f}")
print(f"  unanswerable  min {min(no):.3f}   mean {np.mean(no):.3f}   max {max(no):.3f}")
print(f"  overlap:      {sum(s >= min(yes) for s in no)} unanswerable questions score "
      f"at least as high as the worst answerable one")

best_threshold, best_accuracy = None, 0.0
for threshold in np.arange(0.15, 0.65, 0.005):
    correct = sum((s >= threshold) == a for s, (_, a) in zip(top_scores, QUESTIONS))
    if correct / len(QUESTIONS) > best_accuracy:
        best_threshold, best_accuracy = float(threshold), correct / len(QUESTIONS)
print(f"  the best possible threshold is {best_threshold:.3f}, and it gets "
      f"{best_accuracy:.0%} right")


# ---------------------------------------------------------------- signal 2: ask
class Grounded(BaseModel):
    answer_is_present: bool = Field(
        description="True only if the excerpts contain enough to answer the question "
                    "directly. False if they are merely related to the subject.")
    quote: str | None = Field(
        description="The sentence that answers it, copied exactly. Null if absent.")


def grounded(question: str, excerpts: list[dict]) -> bool:
    context = "\n\n".join(f"[{c['source']}] {c['text']}" for c in excerpts)
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        response_format=Grounded,
        messages=[{"role": "system", "content":
                   "Decide whether the excerpts answer the question. Being about the "
                   "same subject is not enough."},
                  {"role": "user", "content":
                   f"<excerpts>\n{context}\n</excerpts>\n\nQuestion: {question}"}],
    ).choices[0].message.parsed
    return bool(parsed and parsed.answer_is_present)


with ThreadPoolExecutor(max_workers=12) as pool:
    decisions = list(pool.map(lambda pair: grounded(pair[0][0], pair[1]),
                              zip(QUESTIONS, retrieved)))

correct = sum(d == a for d, (_, a) in zip(decisions, QUESTIONS))
false_yes = [q for d, (q, a) in zip(decisions, QUESTIONS) if d and not a]
false_no = [q for d, (q, a) in zip(decisions, QUESTIONS) if not d and a]

print(f"\nSignal 2 — asking the model whether the excerpts answer it\n")
print(f"  correct       {correct}/{len(QUESTIONS)}  {correct / len(QUESTIONS):.0%}")
print(f"  answered when it should not have  {len(false_yes)}")
for q in false_yes:
    print(f"      {q}")
print(f"  refused when it could have        {len(false_no)}")
for q in false_no:
    print(f"      {q}")

Path("code/13/_abstention.json").write_text(json.dumps({
    "k": K,
    "scores": [{"question": q, "answerable": a, "top_score": s, "grounded": bool(d)}
               for (q, a), s, d in zip(QUESTIONS, top_scores, decisions)],
    "threshold_best": {"value": best_threshold, "accuracy": best_accuracy},
    "grounded_accuracy": correct / len(QUESTIONS),
}, indent=2))

print()
print("The scores overlap, so no threshold separates the two groups — and this is not")
print("a tuning problem. Chapter 11 explained why: similarity ranks candidates against")
print("each other, and something always ranks first.")
print()
print("Reading the retrieved text and asking whether it actually answers the question")
print("is a different kind of signal, and it costs one extra call.")
