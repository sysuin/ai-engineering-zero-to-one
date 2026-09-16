# timeout: 1200
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# The same two prompts, sent to both models behind the gateway's settings. What changes when
# only the model does: how much each writes, whether it follows an exact refusal format, and
# whether the book's refusal detector still recognises its refusals.

import re
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST, MODEL_SMART             # noqa: E402
from clarity.evals.runner import abstained, correct, load      # noqa: E402

client = OpenAI()
REVIEWS = Path("data/meridian/documents/quarterly-reviews")
MARKER = "NOT IN THE DOCUMENT"
PROMPTS = {
    "exact": ("Answer in one sentence, using only the document. If the "
              "document does not contain the answer, reply with exactly: "
              f"{MARKER}"),
    "loose": ("Answer using only the document. If the document does not "
              "contain the answer, say so."),
}
cases = ([c for c in load() if c["kind"] == "document" and c["source"].startswith("qbr-")][:12]
         + [c for c in load() if c["kind"] == "unanswerable"][:8])


def document_for(case: dict) -> str:
    return (REVIEWS / (case["source"] or "qbr-2024-Q3.md")).read_text()


def ask(model: str, prompt: str, case: dict) -> str:
    reply = client.chat.completions.create(
        model=model, max_completion_tokens=2_000,
        messages=[{"role": "system", "content": prompt},
                  {"role": "user", "content": f"{document_for(case)}\n\nQuestion: {case['question']}"}])
    return reply.choices[0].message.content or ""


def sentences(text: str) -> int:
    return len([s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s])


answerable = [c for c in cases if c["kind"] != "unanswerable"]
unanswerable = [c for c in cases if c["kind"] == "unanswerable"]
print(f"{len(answerable)} answerable and {len(unanswerable)} unanswerable questions, "
      "each with one quarterly review as context\n")
print(f"  {'model':<8}{'prompt':<8}{'right':>7}{'words':>7}{'> 1 sentence':>14}"
      f"{'refusals seen':>15}{'exact marker':>14}")
missed: dict[tuple[str, str], list[tuple[str, str]]] = {}
for label, model in (("fast", MODEL_FAST), ("smart", MODEL_SMART)):
    for name, prompt in PROMPTS.items():
        with ThreadPoolExecutor(max_workers=10) as pool:
            replies = list(pool.map(lambda c: ask(model, prompt, c), cases))
        by_id = dict(zip((c["id"] for c in cases), replies))
        right = sum(correct(c, by_id[c["id"]]) for c in answerable)
        words = statistics.median(len(by_id[c["id"]].split()) for c in answerable)
        long = sum(sentences(by_id[c["id"]]) > 1 for c in answerable)
        seen = sum(abstained(by_id[c["id"]]) for c in unanswerable)
        exact = sum(by_id[c["id"]].strip().rstrip(".") == MARKER for c in unanswerable)
        print(f"  {label:<8}{name:<8}{right:>4}/{len(answerable)}{words:>7.0f}"
              f"{long:>11}/{len(answerable)}{seen:>12}/{len(unanswerable)}"
              f"{(f'{exact}/{len(unanswerable)}' if name == 'exact' else '-'):>14}")
        missed[(label, name)] = [(c["question"], by_id[c["id"]]) for c in unanswerable
                                 if not abstained(by_id[c["id"]])]

for (label, name), rows in missed.items():
    for question, reply in rows:
        print(f"\n  {label}, {name}: not recognised as a refusal")
        print(f"    Q: {question}")
        print(f"    A: {' '.join(reply.split())[:150]}")
