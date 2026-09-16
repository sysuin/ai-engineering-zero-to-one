# timeout: 3000
# Depends on clarity/evals/runner.py.
# Four optimisations, applied in order, with the eval score beside the price.

import json
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST, rate                    # noqa: E402
from clarity.evals.runner import correct, load                 # noqa: E402
from clarity.platform.cache import ExactCache                  # noqa: E402
from clarity.platform.gateway import Gateway, OpenAIProvider   # noqa: E402
from clarity.v0_5.rag import Clarity                           # noqa: E402
from meridian_index import load_index                          # noqa: E402

key = MODEL_FAST.upper().replace("-", "_").replace(".", "_")
placeholder = rate(MODEL_FAST) is None
if placeholder:
    os.environ[f"RATE_{key}_INPUT"] = "0.15"
    os.environ[f"RATE_{key}_OUTPUT"] = "0.60"
RATES = rate(MODEL_FAST)

VERBOSE = ("You are an analyst for Meridian. Answer the question using only the "
           "passages provided. Explain your reasoning, cite the passages you used, "
           "and note anything the passages do not settle.")
TERSE = ("Answer from the passages only. One sentence. State the figure or name and "
         "nothing else. If the passages do not contain it, say so in one sentence.")

chunks, vectors = load_index()
retriever = Clarity(chunks, vectors)
cases = ([c for c in load() if c["kind"] == "document"][:20] +
         [c for c in load() if c["kind"] == "unanswerable"][:10])
# Real traffic repeats. A fifth of these questions are asked twice, which is the
# conservative end of what a support-style workload looks like.
traffic = cases + cases[:6]

context_cache = {}
with ThreadPoolExecutor(max_workers=6) as pool:
    for case, passages in zip(cases, pool.map(
            lambda c: retriever.retrieve(c["question"], k=8), cases)):
        context_cache[case["id"]] = [p.text for p in passages]

gateway = Gateway(OpenAIProvider(MODEL_FAST))


def measure(label: str, system: str, passages: int, max_tokens: int,
            cache: ExactCache | None) -> dict:
    tokens_in = tokens_out = 0
    calls = 0
    seconds = []
    right = 0
    scored = []

    def ask(case):
        nonlocal tokens_in, tokens_out, calls
        context = "\n\n".join(context_cache[case["id"]][:passages])
        if cache is not None:
            hit = cache.get(case["question"], n=passages, s=system[:20])
            if hit is not None:
                return case, hit, 0.0
        started = time.perf_counter()
        reply = gateway.complete(
            [{"role": "system", "content": system},
             {"role": "user", "content": f"Passages:\n{context[:8000]}\n\n"
                                         f"Question: {case['question']}"}],
            max_tokens=max_tokens)
        elapsed = time.perf_counter() - started
        tokens_in += reply.tokens_in
        tokens_out += reply.tokens_out
        calls += 1
        if cache is not None:
            cache.put(case["question"], reply.text, n=passages, s=system[:20])
        return case, reply.text, elapsed

    # Sequential on purpose: the cache only helps if the first ask has finished
    # before the repeat arrives, which is also true of production.
    for case in traffic:
        case, answer, elapsed = ask(case)
        if elapsed:
            seconds.append(elapsed)
        ok = correct(case, answer)
        right += ok
        scored.append(ok)

    cost = (tokens_in * RATES[0] + tokens_out * RATES[1]) / 1e6
    return {"label": label, "cost": cost, "calls": calls,
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "correct": right, "of": len(traffic),
            "p50": statistics.median(seconds) if seconds else 0.0,
            "cache_rate": cache.rate if cache else 0.0}


steps = [
    ("baseline", VERBOSE, 8, 700, None),
    ("+ ask for one sentence", TERSE, 8, 700, None),
    ("+ cap the output", TERSE, 8, 120, None),
    ("+ trim the context to 4", TERSE, 4, 120, None),
    ("+ exact cache", TERSE, 4, 120, ExactCache()),
]

print(f"{len(traffic)} requests ({len(cases)} distinct; the rest are repeats).\n")
print(f"  {'':<26}{'cost':>10}{'calls':>7}{'out tok':>9}{'p50':>8}{'correct':>10}")
results = []
for label, system, passages, max_tokens, cache in steps:
    row = measure(label, system, passages, max_tokens, cache)
    results.append(row)
    print(f"  {label:<26}{row['cost']:>10.5f}{row['calls']:>7}"
          f"{row['tokens_out']:>9,}{row['p50']:>7.1f}s"
          f"{row['correct']:>6}/{row['of']:<4}")

json.dump({"steps": results, "placeholder": placeholder},
          open("code/28/_case_study.json", "w"), indent=1)

base, final = results[0], results[-1]
saved = 1 - final["cost"] / base["cost"]
quality = (final["correct"] - base["correct"]) / base["of"]
print()
print(f"  cost      {saved:.0%} lower")
print(f"  quality   {base['correct']}/{base['of']} -> {final['correct']}/"
      f"{final['of']}  ({quality:+.0%})")
print(f"  latency   {base['p50']:.1f}s -> {final['p50']:.1f}s median")
if placeholder:
    print()
    print("  (placeholder rates; the ratios are real, the dollars are illustrative)")

print()
print("Both numbers, on the same line, every time. A cheaper system that is quietly")
print("worse is not an optimisation — it is a regression with a business case, and")
print("the only thing that tells them apart is the right-hand column.")
print()
print("Read the steps in order of what they bought.")
print()
print("Asking for one sentence is the largest single win here and the least")
print("fashionable. Output tokens are a minority of the volume at several times the")
print("price, and the default behaviour of every model is to explain itself.")
print()
print("Capping max_tokens is not the same lever. It truncates rather than shortens,")
print("which produces a cut-off answer at full price — §27.1's stop reason is how you")
print("find out that is happening. It is a safety rail, not an optimisation.")
print()
print("Trimming the context is the one to watch. Fewer passages is less input and")
print("fewer chances to retrieve the right one, so it trades cost against §22.10's")
print("context recall directly. Measure both or you will trade away the answer.")
print()
print("And the cache is free money in proportion to how much your traffic repeats,")
print("which is a property of your users rather than your code. Measure the repeat")
print("rate before building anything.")
