# timeout: 2400
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# The same suite, two models, one table.

import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST, MODEL_SMART, rate       # noqa: E402
from clarity.evals.runner import correct, load                 # noqa: E402
from clarity.platform.gateway import Gateway, OpenAIProvider   # noqa: E402
from clarity.v0_5.rag import Clarity                           # noqa: E402
from meridian_index import load_index                          # noqa: E402

K = 6
SYSTEM = ("Answer using only the passages given. Be brief. If the passages do not "
          "contain the answer, say so plainly.")

chunks, vectors = load_index()
retriever = Clarity(chunks, vectors)
cases = ([c for c in load() if c["kind"] == "document"][:18] +
         [c for c in load() if c["kind"] == "unanswerable"][:12])

# The same retrieved context for both models, fetched once. Otherwise the comparison
# includes retrieval variance and stops being about the models at all.
context = {}
with ThreadPoolExecutor(max_workers=6) as pool:
    for case, passages in zip(cases, pool.map(
            lambda c: retriever.retrieve(c["question"], k=K), cases)):
        context[case["id"]] = "\n\n".join(p.text for p in passages)


def run(model: str) -> dict:
    gateway = Gateway(OpenAIProvider(model))

    def ask(case):
        started = time.perf_counter()
        reply = gateway.complete(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Passages:\n{context[case['id']][:6000]}"
                                         f"\n\nQuestion: {case['question']}"}],
            max_tokens=250)
        return case, reply, time.perf_counter() - started

    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(ask, cases))

    right = sum(correct(case, reply.text) for case, reply, _ in rows)
    refusals = sum(1 for case, reply, _ in rows
                   if case["kind"] == "unanswerable" and correct(case, reply.text))
    seconds = sorted(s for _, _, s in rows)
    rates = rate(model)
    tokens_in = sum(r.tokens_in for _, r, _ in rows)
    tokens_out = sum(r.tokens_out for _, r, _ in rows)
    cost = ((tokens_in * rates[0] + tokens_out * rates[1]) / 1e6) if rates else None
    return {"model": model, "correct": right, "of": len(cases),
            "refusals": refusals,
            "refusals_of": sum(1 for c in cases if c["kind"] == "unanswerable"),
            "p50": statistics.median(seconds),
            "p95": seconds[int(0.95 * (len(seconds) - 1))],
            "tokens_in": tokens_in, "tokens_out": tokens_out, "cost": cost}


print(f"{len(cases)} cases, the same retrieved context for both, two models.\n")
print(f"  {'':<16}{'correct':>10}{'refused':>10}{'p50':>9}{'p95':>9}"
      f"{'out tokens':>12}")
results = {}
for model in (MODEL_FAST, MODEL_SMART):
    row = run(model)
    results[model] = row
    print(f"  {model:<16}{row['correct']:>4}/{row['of']:<5}"
          f"{row['refusals']:>4}/{row['refusals_of']:<5}"
          f"{row['p50']:>8.1f}s{row['p95']:>8.1f}s{row['tokens_out']:>12,}")

json.dump(results, open("code/27/_bench.json", "w"), indent=1)
fast, smart = results[MODEL_FAST], results[MODEL_SMART]

print()
if fast["cost"] is not None and smart["cost"] is not None:
    print(f"  cost for the batch   {MODEL_FAST}: ${fast['cost']:.4f}   "
          f"{MODEL_SMART}: ${smart['cost']:.4f}")
else:
    print("  (no rates set, so this table reports tokens and stays silent about")
    print("   dollars — Appendix C says where to find current prices)")

print()
print("This is the table the gateway exists to make possible, and it is worth being")
print("precise about what it does and does not say.")
print()
print(f"It says these two models score {fast['correct']}/{fast['of']} and "
      f"{smart['correct']}/{smart['of']} on this suite, at these")
print("latencies, for these token counts. That is a fact about Meridian's questions")
print("and Meridian's documents, measured on a Tuesday.")
print()
print("It does not say which model is better. Nothing does. There is no such number,")
print("and the published leaderboards answer a question nobody in your building is")
print("asking — Chapter 21 spent forty pages on why your own cases are the only ones")
print("that transfer.")
print()
print("Two things to check before trusting any row of this table:")
print()
print("  the context was held fixed   both models saw identical passages, fetched")
print("                               once. Re-running retrieval per model measures")
print("                               retrieval variance and calls it a model")
print("                               difference")
print("  the interval is wide         §21.6's arithmetic applies here unchanged; on")
print("                               thirty cases a two-point gap is nothing")
print()
print("And a provider comparison is not only a quality comparison. Chapter 27's last")
print("section is about the fact that compliance, data residency and contractual")
print("terms routinely decide this before capability gets a vote — and they are much")
print("harder to change later than a model name in a config file.")
