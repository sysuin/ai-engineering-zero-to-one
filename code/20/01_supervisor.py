# timeout: 900
# One question, three agents, and the whole conversation between them.
# (Re-run when the semantic layer's revenue note was corrected to net of discount.)

import sys

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_12.team import Team                      # noqa: E402
from meridian_index import load_index                    # noqa: E402

QUESTION = ("What was revenue in 2024 Q3, and what does the company give as the "
            "reason the Midwest fell?")

chunks, vectors = load_index()
team = Team(build_tools(Retriever(chunks, vectors), Warehouse()))

print(f"Q: {QUESTION}\n")
run = team.run(QUESTION)
for line in run.transcript:
    who, rest = line.split(" ", 1)
    print(f"  {who:<11}{rest}")

print(f"\nA: {run.answer.strip()[:280]}")
print(f"\n{run.calls} model calls, {run.tokens:,} tokens, {run.seconds:.1f}s")
print(f"{len(run.delegations)} delegations: "
      f"{', '.join(who for who, _ in run.delegations)}")
print()
print("Read the two questions the supervisor wrote, not the answer. It had to turn one")
print("user question into two self-contained ones, because neither colleague can see")
print("the original and neither can see the other's reply.")
print()
print("That rewriting is the whole design, and it is also the whole risk: everything")
print("the supervisor fails to put in those questions is information the workers do")
print("not have, and everything it puts in badly is a question they will answer well")
print("and uselessly.")
