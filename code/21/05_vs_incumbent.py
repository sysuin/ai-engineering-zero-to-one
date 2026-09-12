# timeout: 3600
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# Reads code/21/_scorecard.json, written by 01. Re-run that first if the golden
# set or the scorer changes — this file grades the answers 01 recorded.
# 84% of what? The number only means something next to another number.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.runner import Scorecard, load                # noqa: E402
from clarity.v0_5.rag import Clarity                            # noqa: E402
from meridian_index import load_index                           # noqa: E402

chunks, vectors = load_index()
incumbent = Clarity(chunks, vectors)
cases = load()


def ask(case: dict) -> tuple[dict, str]:
    # v0.5: retrieve, check grounding, answer. No warehouse, no tools, no loop —
    # which is to say, what Clarity was before Part IV, and what most RAG systems
    # in production still are.
    return case, incumbent.answer(case["question"]).text


card = Scorecard()
with ThreadPoolExecutor(max_workers=8) as pool:
    for case, reply in pool.map(ask, cases):
        card.add(case, reply)

current = json.load(open("code/21/_scorecard.json"))
now = {r["id"]: r["correct"] for r in current}

print(f"Clarity v0.5 (retrieval only) against v0.12 (tools and a loop), "
      f"{len(cases)} cases\n")
print(f"  {'':<16}{'v0.5':>8}{'v0.12':>9}   change")
kinds = card.by("kind")
for kind, (right, total) in kinds.items():
    after = sum(now[r["id"]] for r in card.rows if r["kind"] == kind)
    delta = (after - right) / total
    print(f"  {kind:<16}{right / total:>7.0%}{after / total:>9.0%}   "
          f"{delta:>+6.0%}")
before_all = sum(r["correct"] for r in card.rows)
after_all = sum(now.values())
print(f"  {'overall':<16}{before_all / len(cases):>7.0%}"
      f"{after_all / len(cases):>9.0%}   "
      f"{(after_all - before_all) / len(cases):>+6.0%}")

json.dump({"incumbent": {r["id"]: r["correct"] for r in card.rows},
           "current": now}, open("code/21/_incumbent.json", "w"), indent=1)

gained = [r["id"] for r in card.rows if not r["correct"] and now[r["id"]]]
lost = [r["id"] for r in card.rows if r["correct"] and not now[r["id"]]]
print()
print(f"  cases the newer system fixed : {len(gained)}")
print(f"  cases it broke               : {len(lost)}")
if lost:
    print(f"    {', '.join(lost[:6])}")

print()
print("This is the comparison that survives contact with a manager, and the absolute")
print(f"number is not it. Nobody can tell you whether {after_all / len(cases):.0%} is "
      f"good. Everybody can tell you")
print("whether it beats what you had, by how much, and on which kinds of question.")
print()
print(f"Then read the second list. {len(gained)} cases got better and {len(lost)} got "
      f"worse, and the")
print("average contains both. Three of the regressions are refusals: the older system")
print("said it did not know, and the newer one, holding a warehouse and four tools,")
print("found something to say instead.")
print()
print("That is a real cost of Part IV, it is invisible in a single number, and it is")
print("the kind of thing an eval set exists to catch.")
