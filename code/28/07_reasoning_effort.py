# timeout: 2400
# Reasoning effort is a cost and latency lever that needs no prompt change: the same
# questions, the same retrieved passages, the same model, four settings.

import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from openai import OpenAI                                          # noqa: E402

from clarity.config import MODEL_FAST                              # noqa: E402
from clarity.evals.runner import correct, load                     # noqa: E402
from clarity.v0_5.rag import Clarity                               # noqa: E402
from meridian_index import load_index                              # noqa: E402

EFFORTS, RUNS, K = ("none", "low", "medium", "high"), 2, 6
SYSTEM = ("Answer using only the passages given. Be brief. If the passages do not "
          "contain the answer, say so plainly.")
client = OpenAI()
chunks, vectors = load_index()
retriever = Clarity(chunks, vectors)
cases = ([c for c in load() if c["kind"] == "document"][:18] +
         [c for c in load() if c["kind"] == "unanswerable"][:12])
context = {c["id"]: "\n\n".join(p.text for p in retriever.retrieve(c["question"], k=K))
           for c in cases}


def ask(case: dict, effort: str) -> tuple[bool, int, int, float, str]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_FAST, reasoning_effort=effort, max_completion_tokens=4_000,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Passages:\n{context[case['id']][:6000]}"
                                              f"\n\nQuestion: {case['question']}"}])
    usage = response.usage
    reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0
    text = response.choices[0].message.content or ""
    return (correct(case, text), usage.completion_tokens, reasoning,
            time.perf_counter() - started, response.choices[0].finish_reason)


print(f"{len(cases)} cases, identical passages, {RUNS} runs per setting\n")
print(f"  {'effort':<8} {'correct':>11} {'output tok':>11} {'reasoning':>10} "
      f"{'p50':>6} {'p95':>6}")
summary = {}
for effort in EFFORTS:
    scores, outputs, reasoning, seconds = [], 0, 0, []
    for _ in range(RUNS):
        with ThreadPoolExecutor(max_workers=6) as pool:
            rows = list(pool.map(lambda c: ask(c, effort), cases))
        scores.append(sum(r[0] for r in rows))
        outputs += sum(r[1] for r in rows)
        reasoning += sum(r[2] for r in rows)
        seconds += [r[3] for r in rows]
    seconds.sort()
    summary[effort] = (scores, outputs / RUNS, reasoning / RUNS, statistics.median(seconds))
    runs = " / ".join(str(s) for s in scores)
    print(f"  {effort:<8} {runs:>7} /{len(cases):<3} {outputs / RUNS:>11,.0f} "
          f"{reasoning / RUNS:>10,.0f} {statistics.median(seconds):>5.1f}s "
          f"{seconds[int(0.95 * (len(seconds) - 1))]:>5.1f}s")

low, high = summary[EFFORTS[0]], summary[EFFORTS[-1]]
spread = max(max(s[0]) - min(s[0]) for s in summary.values())
best = max(max(s[0]) for s in summary.values())
worst = min(min(s[0]) for s in summary.values())
print(f"\nFrom '{EFFORTS[0]}' to '{EFFORTS[-1]}', output tokens went from {low[1]:,.0f} to "
      f"{high[1]:,.0f} a run and the")
print(f"median call from {low[3]:.1f}s to {high[3]:.1f}s. Scores ranged from {worst} to {best} "
      f"across every setting and run,")
print(f"and the same setting moved by up to {spread} between its two runs — so on these")
print("questions, read the score column for any difference larger than that, and only then")
print("pay for more thinking.")
