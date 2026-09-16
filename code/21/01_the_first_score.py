# timeout: 3600
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# Clarity, on 120 cases it has never seen. The first honest number.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.runner import Scorecard, load                # noqa: E402
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer "
          "is not in the documents or the warehouse, say so plainly rather than "
          "guessing.")

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())
cases = load()


def answer(case: dict) -> tuple[dict, str, int]:
    run = Agent(tools, budget=Budget(steps=6)).run(case["question"], system=SYSTEM)
    return case, run.answer, run.tokens


card = Scorecard()
with ThreadPoolExecutor(max_workers=8) as pool:
    for case, reply, tokens in pool.map(answer, cases):
        card.add(case, reply, tokens=tokens)

print(f"Clarity, {len(cases)} golden cases it has never seen\n")
print(f"  overall  {card.score:.0%}\n")
print("by kind")
print(card.table("kind"))
print("\nby tier")
print(card.table("tier"))

# "tail" includes the twenty refusals, which Clarity is good at, so comparing head
# with tail flatters the tail. The honest comparison is generated-and-answerable
# against written-and-answerable.
for row in card.rows:
    row["group"] = ("must refuse" if row["kind"] == "unanswerable"
                    else "generated" if row["tier"] == "head" else "written by hand")
print("\nby where the case came from")
print(card.table("group"))

json.dump(card.rows, open("code/21/_scorecard.json", "w"), indent=1)

groups = card.by("group")
generated = groups["generated"]
written = groups["written by hand"]
gap = generated[0] / generated[1] - written[0] / written[1]
print()
print(f"Overall is {card.score:.0%}, and overall is the least useful line on the page.")
print()
if abs(gap) < 0.05:
    print(f"The generated cases and the hand-written ones came out within "
          f"{abs(gap):.0%} of each")
    print("other, which is not what usually happens and is worth saying plainly. It")
    print("does not mean the split was pointless — it means this corpus is unusually")
    print("regular, and the hand-written cases were drawn from the same few documents.")
else:
    print(f"{generated[0] / generated[1]:.0%} on cases a script generated, "
          f"{written[0] / written[1]:.0%} on cases somebody sat and wrote.")
    print("The demos in the first twenty chapters were all the first kind.")
print()
worst = min(card.by("kind").items(), key=lambda kv: kv[1][0] / kv[1][1])
print(f"The line that should bother you is {worst[0]}: {worst[1][0]} of {worst[1][1]}.")
print("That is the number a plan can be built on, and it is invisible in the average.")

# golden.yaml v2: exact warehouse figures accepted alongside the rounded ones
