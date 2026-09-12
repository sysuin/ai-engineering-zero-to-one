# timeout: 1200
# Clarity v0.8: four tools, one loop.

import sys

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever            # noqa: E402
from clarity.v0_7.warehouse import Warehouse           # noqa: E402
from clarity.v0_8.tools import ToolRunner, build_tools  # noqa: E402
from meridian_index import load_index                   # noqa: E402

chunks, vectors = load_index()
runner = ToolRunner(build_tools(Retriever(chunks, vectors), Warehouse()))

SYSTEM = ("You answer questions for Meridian's analytics team. Use tools for anything "
          "factual. Never state a figure you did not obtain from a tool. Cite the "
          "source of anything you read from a document.")

QUESTIONS = [
    "What was revenue in 2024 Q3, and what does the quarterly review say caused the "
    "Midwest decline?",
    "What was gross profit in 2024 Q3 as a percentage of revenue? Compute it.",
    "How many people work at the Columbus depot?",
]

for question in QUESTIONS:
    answer, trace = runner.run(question, SYSTEM)
    print(f"Q: {question[:74]}")
    for turn in trace:
        mark = "FAILED" if turn.failed else "ok"
        args = str(turn.arguments)[:52]
        print(f"   {mark:7} {turn.tool:18} {turn.seconds:5.2f}s  {args}")
        if turn.failed:
            print(f"           -> {turn.result[:60]}")
    print(f"   A: {answer[:260]}")
    print()

print("Three things in those traces are worth reading.")
print()
print("The first question needed both tools, and it used each for what it is for: the")
print("exact figure from the warehouse, the explanation from the documents. That split")
print("is the whole of Parts III and IV in one answer.")
print()
print("The second was asked to 'compute' a percentage, and it did not compute anything.")
print("It recognised that margin_pct is already a defined metric and asked the")
print("warehouse for it — so the number came from Chapter 15's semantic layer rather")
print("than from arithmetic over two figures the model had to keep straight. The best")
print("tool call is often the one that makes a second call unnecessary.")
print()
print("The third has no answer anywhere in the corpus. It searched twice, found the")
print("depot mentioned in three documents, and refused — naming what it did find so a")
print("person can judge whether to look elsewhere.")
