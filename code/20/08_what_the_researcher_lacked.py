# timeout: 1800
# The researcher missed it twice in twelve. Was it missing the other half?

import json
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/20")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from clarity.v0_12.team import RESEARCHER                # noqa: E402
from meridian_index import load_index                    # noqa: E402

RUNS = 12
# The sub-question the supervisor actually wrote, taken verbatim from §20.11's runs.
BARE = "According to the company documents, what reason is given for why the Midwest fell?"
# The same question, with the one thing the analyst already knew.
WITH_CONTEXT = (BARE + "\n\nContext already established: total revenue in 2024 Q3 "
                       "was $8,461,841.81.")

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())
researcher = [t for t in tools if t.name in ("search_documents", "today")]


def trial(question: str, label: str) -> dict:
    hits = 0
    for _ in range(RUNS):
        run = Agent(researcher, budget=Budget(steps=4)).run(question,
                                                            system=RESEARCHER)
        hits += "halloway" in run.answer.lower()
    print(f"  {label:<34} {hits:>2} of {RUNS}")
    return {"label": label, "hits": hits, "runs": RUNS}


print(f"The researcher alone, {RUNS} runs each, asked the same question two ways.\n")
rows = [trial(BARE, "the sub-question as written"),
        trial(WITH_CONTEXT, "with the analyst's figure attached")]
json.dump(rows, open("code/20/_context.json", "w"), indent=1)

bare, ctx = rows[0]["hits"], rows[1]["hits"]
print()
if ctx > bare:
    print(f"Adding one sentence the analyst already had moved it from {bare} to {ctx}.")
    print("The researcher was not worse at reading. It was working without the half")
    print("of the question its colleague was holding.")
else:
    print(f"It did not help: {bare} against {ctx}. Whatever the researcher was")
    print("missing, it was not the analyst's figure — which rules out the tidiest")
    print("explanation and leaves the loss where §20.4 found it, inside retrieval.")
print()
print("Either way, note what this experiment is: the fix for an agent boundary is to")
print("pass more of the context across it. Do that thoroughly enough and you have")
print("rebuilt the single agent, with extra network hops.")
