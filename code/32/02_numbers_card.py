# timeout: 2400
# The numbers card: what Clarity v1.0 actually does, measured end to end.
#
# Every figure below comes from one run over the 120-case golden set plus the ten
# red-team attacks. Nothing here is an average of averages and nothing is quoted from an
# earlier chapter — a card assembled from numbers measured at different times is a card
# that describes a system that never existed.

import json
import statistics as stats
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST, rate                      # noqa: E402
from clarity.evals import redteam                                # noqa: E402
from clarity.evals.runner import correct, load                   # noqa: E402
from clarity.evals.suite import squeeze                          # noqa: E402
from clarity.v1_0.clarity import Clarity                         # noqa: E402

CASES = load()
engine = Clarity(cache=False, max_concurrent=8)


def answer_one(case: dict) -> dict:
    result = engine.ask(case["question"])
    span = case.get("span")
    # Recall is only meaningful where a document is the only place the answer lives.
    # §22.8's recurring trap: where the gold answer came from is not a claim about
    # which tool should have run, and scoring every case here reports 0% on a healthy
    # system because half of them are answered from the warehouse.
    scorable = bool(span) and case.get("tier") == "tail" and case["kind"] == "document"
    return {
        "id": case["id"], "kind": case["kind"],
        "correct": correct(case, result.text),
        "refused": result.refused,
        "grounded": result.grounded,
        "seconds": result.seconds, "tokens": result.tokens,
        "sources": len(result.sources),
        # Against the tool results, not the filenames. A first version compared the
        # span to `" ".join(result.sources)` and reported 0% recall on a system whose
        # answers were 95% correct — a metric that fails its own control is measuring
        # itself.
        "recalled": (squeeze(span)[:48] in squeeze(result.context)
                     if scorable else None),
        "scorable": scorable,
    }


print(f"Clarity v1.0 over {len(CASES)} golden cases and "
      f"{len(redteam.ATTACKS)} attacks. One run, no cache.\n", flush=True)

with ThreadPoolExecutor(max_workers=8) as pool:
    rows = list(pool.map(answer_one, CASES))

# ------------------------------------------------------------------ red team
def attacked(attack) -> bool:
    """A case passes when the tell is absent. Absence is weak evidence, and §29.6 says
    so — but a tell that fires is conclusive, which is the direction that matters."""
    result = engine.ask(attack.question)
    return not attack.tell.search(result.text)


with ThreadPoolExecutor(max_workers=4) as pool:
    blocked = list(pool.map(attacked, redteam.ATTACKS))

# ------------------------------------------------------------------ the card
by_kind = {}
for row in rows:
    by_kind.setdefault(row["kind"], []).append(row)

latencies = sorted(r["seconds"] for r in rows)
tokens = [r["tokens"] for r in rows]
recallable = [r for r in rows if r["scorable"]]
unanswerable = by_kind.get("unanswerable", [])


def pct(part, whole) -> float:
    return len(part) / len(whole) if whole else 0.0


card = {
    "cases": len(rows),
    "pass_rate": pct([r for r in rows if r["correct"]], rows),
    "grounded": pct([r for r in rows if r["grounded"]], rows),
    "abstention": pct([r for r in unanswerable if r["correct"]], unanswerable),
    "false_refusal": pct([r for r in rows
                          if r["refused"] and r["kind"] != "unanswerable"],
                         [r for r in rows if r["kind"] != "unanswerable"]),
    "recall": pct([r for r in recallable if r["recalled"]], recallable),
    "recall_n": len(recallable),
    "blocked": sum(blocked) / len(blocked),
    "p50": stats.median(latencies),
    "p95": latencies[int(0.95 * len(latencies)) - 1],
    "tokens_mean": stats.mean(tokens),
    "by_kind": {k: {"n": len(v), "pass": pct([r for r in v if r["correct"]], v)}
                for k, v in sorted(by_kind.items())},
}

print("  THE NUMBERS CARD\n")
print(f"  pass rate           {card['pass_rate']:>6.0%}   "
      f"over {card['cases']} verified cases")
for kind, k in card["by_kind"].items():
    print(f"    {kind:<17}{k['pass']:>6.0%}   n={k['n']}")
print(f"  grounded            {card['grounded']:>6.0%}   "
      "an answer with a tool behind it")
print(f"  abstention          {card['abstention']:>6.0%}   "
      f"of {len(unanswerable)} questions with no answer")
print(f"  false refusal       {card['false_refusal']:>6.0%}   "
      "refused when an answer existed")
print(f"  recall              {card['recall']:>6.0%}   "
      f"verified passage returned, n={card['recall_n']}")
# n=14 is small enough that this row is a smoke alarm, not a measurement. It is on the
# card anyway, with its n beside it, because a metric you cannot yet trust is still
# worth watching — and hiding it would make the card look more certain than it is.
print(f"  attacks blocked     {card['blocked']:>6.0%}   "
      f"of {len(redteam.ATTACKS)} red-team attempts")
print(f"  latency p50 / p95   {card['p50']:>5.1f}s / {card['p95']:.1f}s")
print(f"  tokens per answer   {card['tokens_mean']:>6,.0f}")

prices = rate(MODEL_FAST)
if prices:
    # A blended figure: the split between input and output varies per answer, and the
    # card is a summary, not an invoice.
    per_answer = card["tokens_mean"] * (prices[0] * 0.85 + prices[1] * 0.15) / 1e6
    print(f"  cost per answer     ${per_answer:>6.4f}")
    card["cost_per_answer"] = per_answer
else:
    print("  cost per answer       — set RATE_<MODEL>_INPUT/_OUTPUT in .env")

print("\nWhat this card is for, and what it is not.\n")
print("It is a baseline. Every one of these numbers is a measurement of this system on")
print("this corpus today, which means the only honest use of it is comparison: against")
print("next month's card, against the incumbent, against the version before the change")
print("you are about to argue for.")
print()
gap = card["pass_rate"] - card["abstention"]
if card["abstention"] < card["pass_rate"]:
    print(f"Read the weakest row first. Abstention is {card['abstention']:.0%} against "
          f"an overall {card['pass_rate']:.0%} —")
    points = round(abs(gap) * 100)
    print(f"{points} point{'s' if points != 1 else ''} below, and refusing is the "
          "behaviour with the most expensive")
    print("failure mode, because a confident wrong answer is worse than a slow one.")
else:
    points = round(abs(gap) * 100)
    print(f"Read the weakest row first. Abstention is {card['abstention']:.0%}, "
          f"{points} point{'s' if points != 1 else ''} above the overall")
    print(f"{card['pass_rate']:.0%} — the system is more reliable at knowing what it "
          "does not know than")
    print("at knowing what it does, which is the safer direction to be uneven in.")
print()
worst = min(card["by_kind"].items(), key=lambda kv: kv[1]["pass"])
print(f"The weakest population is {worst[0]} at {worst[1]['pass']:.0%} over "
      f"{worst[1]['n']} cases. That is where")
print("the next week of work goes, and the card is how you will know whether it helped.")
print()
print(f"And note the interval. At n={card['cases']}, a difference of three points "
      "between this")
print("card and the next one is noise. §21.6 and Appendix F are how you tell the")
print("difference between an improvement and a good afternoon.")

card["rows"] = rows          # §32.7 triages these into the next eval set
json.dump(card, open("code/32/_card.json", "w"), indent=1)
