# timeout: 1800
# Where in a three-agent pipeline does a fact get lost?

import json
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/20")
from _taskset import TASKS, score                        # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_12.team import Team                      # noqa: E402
from meridian_index import load_index                    # noqa: E402

RUNS = 12
TASK = TASKS[0]                       # revenue in 2024 Q3, and why the Midwest fell
WANTED = TASK["cause"]                # ["halloway"]

chunks, vectors = load_index()
team = Team(build_tools(Retriever(chunks, vectors), Warehouse()))


def mentions(text: str) -> bool:
    return any(word in (text or "").lower() for word in WANTED)


rows = []
for _ in range(RUNS):
    run = team.run(TASK["q"])
    asked = next((q for who, q in run.delegations if who == "researcher"), "")
    # The full reply, not the transcript line — that one is truncated for reading,
    # and an earlier version of this listing scored it, which counted a hit as a miss.
    replied = next((text for who, text in run.replies if who == "researcher"), "")
    rows.append({"asked": asked,
                 "asked_named_period": "q3" in asked.lower(),
                 "researcher_found": mentions(replied),
                 "final_has_it": bool(score(TASK, run.answer)[1])})

found = sum(r["researcher_found"] for r in rows)
final = sum(r["final_has_it"] for r in rows)
lost_in_research = RUNS - found
lost_in_handoff = sum(1 for r in rows
                      if r["researcher_found"] and not r["final_has_it"])

print(f"{RUNS} runs of one question the single agent answered correctly every time.\n")
print(f"  the researcher found the cause      : {found:>2} of {RUNS}")
print(f"  it survived into the final answer   : {final:>2} of {RUNS}")
print()
print(f"  lost because the researcher missed it : {lost_in_research}")
print(f"  lost between the researcher and the answer : {lost_in_handoff}")

named = sum(r["asked_named_period"] for r in rows)
print(f"\n  sub-questions that named the quarter : {named} of {RUNS}")
print(f"  of those, the researcher found it    : "
      f"{sum(r['researcher_found'] for r in rows if r['asked_named_period'])}")
print(f"  of the rest, the researcher found it : "
      f"{sum(r['researcher_found'] for r in rows if not r['asked_named_period'])}")

json.dump({"runs": RUNS, "rows": rows, "found": found, "final": final,
           "lost_in_research": lost_in_research,
           "lost_in_handoff": lost_in_handoff, "named": named},
          open("code/20/_interface.json", "w"), indent=1)

print()
misses = [r for r in rows if not r["researcher_found"]]
hits = [r for r in rows if r["researcher_found"]]
if misses and hits:
    print("A sub-question that worked, and one that did not:\n")
    # Show the tails: these sentences differ at the end, and an earlier version of
    # this listing truncated them into looking identical.
    print(f"  found  : …{hits[0]['asked'][-72:]}")
    print(f"  missed : …{misses[0]['asked'][-72:]}")
    print()

named_hits = sum(r["researcher_found"] for r in rows if r["asked_named_period"])
unnamed = [r for r in rows if not r["asked_named_period"]]
unnamed_hits = sum(r["researcher_found"] for r in unnamed)
print("Two things in this output, and only one of them is stable.\n")
print(f"The unstable one is the quarter. Today {named} of {RUNS} sub-questions carried")
print(f"it, and the split was {named_hits}/{named} against {unnamed_hits}/"
      f"{len(unnamed)}. A previous run of this same listing")
print("produced almost the reverse proportion. The supervisor's phrasing is itself a")
print("sample from a model, so it is not a variable you get to hold still.")
print()
print(f"The stable one is the shape of the loss: {lost_in_handoff} of the misses "
      f"happened between the")
print("researcher and the final answer. Everything the researcher found came back")
print("intact. The loss is inside a specialist that is running the same retrieval,")
print("over the same corpus, with the same tool the single agent used.")
print()
print("Which leaves one obvious suspect. The single agent searched the documents")
print("while holding the figure it had just computed; the researcher searched holding")
print("nothing at all. That is testable, and the next listing tests it.")
