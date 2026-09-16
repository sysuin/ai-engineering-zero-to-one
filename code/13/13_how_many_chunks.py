# timeout: 3600
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# "Increase k until it stops improving", measured end to end: the golden set's document questions
# and its unanswerable ones, answered from the top k excerpts, for five values of k. Recall is free
# to compute; what an answer does with more excerpts is not.

import statistics
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_EMBED, MODEL_FAST              # noqa: E402
from clarity.evals.runner import abstained, correct, load, plain  # noqa: E402
from meridian_index import load_index                           # noqa: E402

KS = (1, 3, 5, 10, 20)
client = OpenAI()
chunks, vectors = load_index()
cases = [c for c in load() if (c["kind"] == "document" and c.get("span")) or c["kind"] == "unanswerable"]
q = np.array([d.embedding for d in client.embeddings.create(
    model=MODEL_EMBED, input=[c["question"] for c in cases]).data], dtype=np.float32)
q /= np.linalg.norm(q, axis=1, keepdims=True)
order = np.argsort(-(q @ vectors.T), axis=1)
SYSTEM = ("Answer only from the excerpts, citing the source of each fact. If the excerpts do not "
          "contain the answer, say so plainly.")


def has_span(case: dict, top: list[int]) -> bool:
    probe = plain(" ".join(case["span"].split())[:60])
    return any(probe in plain(" ".join(chunks[i]["text"].split())) for i in top)


def one(args):
    n, k = args
    case, top = cases[n], order[n][:k].tolist()
    excerpts = "\n\n".join(f"[{chunks[i]['source']}, {chunks[i]['heading']}]\n{chunks[i]['text']}"
                           for i in top)
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"<excerpts>\n{excerpts}\n</excerpts>\n\n"
                                              f"Question: {case['question']}"}])
    text = response.choices[0].message.content or ""
    retrieved = case["kind"] != "unanswerable" and has_span(case, top)
    return k, case["kind"], retrieved, correct(case, text), abstained(text), response.usage.prompt_tokens


with ThreadPoolExecutor(max_workers=12) as pool:
    rows = list(pool.map(one, [(n, k) for n in range(len(cases)) for k in KS]))

answerable = sum(c["kind"] != "unanswerable" for c in cases)
print(f"{answerable} document questions with a verified span, {len(cases) - answerable} unanswerable; "
      f"dense retrieval, labelled excerpts\n")
print(f"  {'k':>3}{'span retrieved':>16}{'answered right':>16}{'right when':>12}{'refused the':>13}{'prompt':>9}")
print(f"  {'':>3}{'':>16}{'':>16}{'retrieved':>12}{'unanswerable':>13}{'tokens':>9}")
for k in KS:
    mine = [r for r in rows if r[0] == k]
    ans = [r for r in mine if r[1] != "unanswerable"]
    una = [r for r in mine if r[1] == "unanswerable"]
    got = [r for r in ans if r[2]]
    print(f"  {k:>3}{sum(r[2] for r in ans) / len(ans):>16.0%}{sum(r[3] for r in ans) / len(ans):>16.0%}"
          f"{sum(r[3] for r in got) / max(1, len(got)):>12.0%}{sum(r[3] for r in una) / len(una):>13.0%}"
          f"{statistics.median(r[5] for r in mine):>9,.0f}")
print("\n'right when retrieved': answers right, of the questions whose verified span was among the k")
