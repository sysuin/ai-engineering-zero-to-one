# timeout: 2400
# Route by difficulty: the cheap model answers first, a check that can fail decides whether its
# answer stands, and only the failures go to the expensive model. Measured against each model
# alone, on the same cases and the same retrieved passages.

import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from openai import OpenAI                                          # noqa: E402
from pydantic import BaseModel                                     # noqa: E402

from clarity.config import MODEL_FAST, MODEL_SMART                 # noqa: E402
from clarity.evals.runner import abstained, correct, load, plain   # noqa: E402
from clarity.v0_5.rag import Clarity                               # noqa: E402
from meridian_index import load_index                              # noqa: E402

K = 6
SYSTEM = ("Answer using only the passages. Give the answer in one sentence, and copy the one "
          "sentence of the passages that supports it into `quote`, exactly. If the passages do "
          "not contain the answer, say so and leave `quote` empty.")
client = OpenAI()
chunks, vectors = load_index()
retriever = Clarity(chunks, vectors)
cases = [c for c in load() if c["kind"] in ("document", "unanswerable")]
context = {c["id"]: "\n\n".join(p.text for p in retriever.retrieve(c["question"], k=K))[:6000]
           for c in cases}


class Reply(BaseModel):
    answer: str
    quote: str


def ask(case: dict, model: str) -> tuple[Reply, int]:
    effort = "none" if model == MODEL_FAST else "low"
    response = client.chat.completions.parse(
        model=model, reasoning_effort=effort, max_completion_tokens=4_000, response_format=Reply,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Passages:\n{context[case['id']]}\n\n"
                                              f"Question: {case['question']}"}])
    return response.choices[0].message.parsed, response.usage.total_tokens


def numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in re.findall(r"\d[\d,]*(?:\.\d+)?", text)}


def stands(case: dict, reply: Reply) -> bool:
    """A refusal stands. An answer stands only if its quote is really in
    the passages and every number in the answer is in the quote."""
    if abstained(reply.answer):
        return True
    quote = plain(" ".join(reply.quote.split()))
    passages = plain(" ".join(context[case["id"]].split()))
    return (len(quote) > 20 and quote in passages
            and numbers(reply.answer) <= numbers(reply.quote))


with ThreadPoolExecutor(max_workers=8) as pool:
    fast = list(pool.map(lambda c: ask(c, MODEL_FAST), cases))
    smart = list(pool.map(lambda c: ask(c, MODEL_SMART), cases))

rows = []
for case, (f, f_tokens), (s, s_tokens) in zip(cases, fast, smart):
    kept = stands(case, f)
    final = f if kept else s
    rows.append({"fast": correct(case, f.answer), "smart": correct(case, s.answer),
                 "cascade": correct(case, final.answer), "escalated": not kept,
                 "fast_tokens": f_tokens, "smart_tokens": s_tokens, "fast_ok_when_kept":
                 correct(case, f.answer) if kept else None})

n = len(rows)
escalated = sum(r["escalated"] for r in rows)
print(f"{n} questions ({sum(c['kind'] == 'unanswerable' for c in cases)} with no answer), "
      f"the same {K} passages for every model\n")
print(f"  {'strategy':26}{'correct':>9}{'fast calls':>12}{'smart calls':>13}"
      f"{'smart tokens':>14}")
smart_tokens = sum(r["smart_tokens"] for r in rows)
esc_tokens = sum(r["smart_tokens"] for r in rows if r["escalated"])
print(f"  {'fast model alone':26}{sum(r['fast'] for r in rows):>5}/{n}{n:>12}{0:>13}{0:>14,}")
print(f"  {'smart model alone':26}{sum(r['smart'] for r in rows):>5}/{n}{0:>12}{n:>13}"
      f"{smart_tokens:>14,}")
print(f"  {'fast, escalate on check':26}{sum(r['cascade'] for r in rows):>5}/{n}{n:>12}"
      f"{escalated:>13}{esc_tokens:>14,}")

kept = [r for r in rows if not r["escalated"]]
wrong_kept = sum(not r["fast"] for r in kept)
caught = sum(1 for r in rows if r["escalated"] and not r["fast"])
print(f"\nthe check kept {len(kept)} fast answers, {wrong_kept} of them wrong; it escalated "
      f"{escalated},")
print(f"of which the fast answer was wrong in {caught} and right in {escalated - caught}")
