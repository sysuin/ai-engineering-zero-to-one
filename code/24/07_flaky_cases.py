# timeout: 2400
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# One run labels every case pass or fail. Ten runs show that some cases are neither: they
# pass some of the time, and a single run sorts them into a bucket by coin toss.

import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from openai import OpenAI                                          # noqa: E402

from clarity.config import MODEL_FAST                              # noqa: E402
from clarity.evals.runner import correct, load                     # noqa: E402
from clarity.v0_5.rag import Clarity                               # noqa: E402
from meridian_index import load_index                              # noqa: E402

RUNS, K = 10, 6
SYSTEM = ("Answer the question using only the passages given. Be brief. If the passages "
          "do not contain the answer, say so.")
client = OpenAI()
chunks, vectors = load_index()
retriever = Clarity(chunks, vectors)
cases = [c for c in load() if c["kind"] == "document"][:24]
context = {c["id"]: "\n\n".join(p.text for p in retriever.retrieve(c["question"], k=K))
           for c in cases}                 # retrieved once: only the model varies below


def passes(case: dict) -> bool:
    reply = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=300,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"Passages:\n{context[case['id']][:6000]}"
                                              f"\n\nQuestion: {case['question']}"}])
    return correct(case, reply.choices[0].message.content or "")


with ThreadPoolExecutor(max_workers=8) as pool:
    rates = {c["id"]: sum(pool.map(passes, [c] * RUNS)) for c in cases}

always = [i for i, r in rates.items() if r == RUNS]
never = [i for i, r in rates.items() if r == 0]
flaky = {i: r for i, r in rates.items() if 0 < r < RUNS}
print(f"{len(cases)} cases, the same passages every time, {RUNS} runs each\n")
print(f"  passed every run    {len(always):>3}")
print(f"  failed every run    {len(never):>3}")
print(f"  flaky               {len(flaky):>3}")
for case_id, r in sorted(flaky.items(), key=lambda kv: kv[1]):
    print(f"    {case_id:<24} passed {r:>2} of {RUNS}")

# What a single run would have reported, on average, for the flaky cases.
expected_fail = sum(1 - r / RUNS for r in flaky.values())
print(f"\nA single run would call about {expected_fail:.1f} of the flaky cases failures, and "
      "a different")
print("handful on each run. Triage those as bugs and some fixes are fixes for noise;")
print("ignore them and the cases that fail most of the time go unexamined. Label a case by")
print("its rate over several runs, and spend the debugging time on the ones that fail most.")
