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
from clarity.evals.runner import abstained, correct, load      # noqa: E402
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


RUNS = 2
# The smart model twice: once with the same output ceiling as the fast one, and once
# with room. A reasoning model spends output tokens before it writes anything visible,
# so one ceiling for both is not the same condition for both.
ARMS = [(MODEL_FAST, 250), (MODEL_SMART, 250), (MODEL_SMART, 2_000)]


def run(model: str, ceiling: int) -> dict:
    gateway = Gateway(OpenAIProvider(model))

    def ask(case):
        started = time.perf_counter()
        reply = gateway.complete(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": f"Passages:\n{context[case['id']][:6000]}"
                                         f"\n\nQuestion: {case['question']}"}],
            max_tokens=ceiling)
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
    return {"model": model, "ceiling": ceiling, "correct": right, "of": len(cases),
            "refusals": refusals,
            "refusals_of": sum(1 for c in cases if c["kind"] == "unanswerable"),
            "cut_off": sum(1 for _, r, _ in rows if r.stop_reason == "length"),
            "empty": sum(1 for _, r, _ in rows if not r.text.strip()),
            "p50": statistics.median(seconds),
            "p95": seconds[int(0.95 * (len(seconds) - 1))],
            "tokens_in": tokens_in, "tokens_out": tokens_out,
            "reasoning": sum(r.tokens_reasoning for _, r, _ in rows), "cost": cost,
            # Per case, for §27.5's routing arithmetic.
            "cases": [{"id": case["id"], "kind": case["kind"],
                       "correct": correct(case, reply.text),
                       "abstained": abstained(reply.text),
                       "tokens_in": reply.tokens_in, "tokens_out": reply.tokens_out}
                      for case, reply, _ in rows]}


print(f"{len(cases)} cases, the same retrieved context for every arm, {RUNS} runs each.\n")
print(f"  {'':<16}{'ceiling':>8}{'correct':>9}{'refused':>9}{'cut off':>8}"
      f"{'p50':>7}{'out tokens':>11}{'reasoning':>10}")
results = []
for model, ceiling in ARMS:
    for attempt in range(RUNS):
        row = run(model, ceiling)
        results.append(row)
        print(f"  {model if attempt == 0 else '':<16}{ceiling:>8,}"
              f"{row['correct']:>5}/{row['of']:<3}{row['refusals']:>5}/{row['refusals_of']:<3}"
              f"{row['cut_off']:>8}{row['p50']:>6.1f}s{row['tokens_out']:>11,}"
              f"{row['reasoning']:>10,}")

json.dump(results, open("code/27/_bench.json", "w"), indent=1)


def arm(model, ceiling):
    return [r for r in results if r["model"] == model and r["ceiling"] == ceiling]


fast, tight, roomy = arm(MODEL_FAST, 250), arm(MODEL_SMART, 250), arm(MODEL_SMART, 2_000)
spread = max(abs(a["correct"] - b["correct"]) for a, b in (fast, tight, roomy))

print()
if all(r["cost"] is not None for r in results):
    for r in results:
        print(f"  cost   {r['model']} at {r['ceiling']:,}: ${r['cost']:.4f}")
else:
    print("  (no rates set, so this table reports tokens and stays silent about")
    print("   dollars — Appendix C says where to find current prices)")

print()
cut = sum(r["cut_off"] for r in tight)
empty = sum(r["empty"] for r in tight)
if cut:
    print(f"Read the cut-off column first. At a {tight[0]['ceiling']} token ceiling the smart "
          f"model stopped at")
    print(f"'length' {cut} time{'s' if cut != 1 else ''} across {RUNS} runs, {empty} of them "
          "with no visible text at all: its")
    print("reasoning tokens count against the same ceiling as its answer. Scored as it")
    print("stands, that arm partly measures max_tokens rather than the model.")
else:
    print(f"The cut-off column is empty this time: at a {tight[0]['ceiling']} token ceiling "
          "neither model")
    print("stopped at 'length'. Reasoning tokens count against that ceiling, so check the")
    print("column on every run rather than assuming it stays empty.")
print()
print(f"Then the runs. The same arm scored up to {spread} "
      f"case{'s' if spread != 1 else ''} apart from one run to the next.")
print("A gap between models smaller than that is not a finding.")
print()
print("This is the table the gateway exists to make possible, and it is worth being")
print("precise about what it does and does not say.")
print()
print("It says how these arms scored on this suite, at these latencies, for these token")
print("counts. That is a fact about Meridian's questions and Meridian's documents,")
print("measured on one afternoon.")
print()
print("It does not say which model is better. Nothing does. There is no such number,")
print("and the published leaderboards answer a question nobody in your building is")
print("asking — Chapter 21 is about why your own cases are the only ones")
print("that transfer.")
print()
print("Three things to check before trusting any row of this table:")
print()
print("  the context was held fixed   both models saw identical passages, fetched")
print("                               once. Re-running retrieval per model measures")
print("                               retrieval variance and calls it a model")
print("                               difference")
print("  the ceiling was fair         one output ceiling is not one condition when")
print("                               one model reasons before it answers")
print("  the interval is wide         §21.6's arithmetic applies here unchanged; on")
print("                               thirty cases a gap of a few cases is nothing")
print()
print("And a provider comparison is not only a quality comparison. This chapter's last")
print("section is about the fact that compliance, data residency and contractual")
print("terms routinely decide this before capability gets a vote — and they are much")
print("harder to change later than a model name in a config file.")
