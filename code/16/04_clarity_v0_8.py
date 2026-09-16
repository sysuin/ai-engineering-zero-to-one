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

traces = []
for question in QUESTIONS:
    answer, trace = runner.run(question, SYSTEM)
    traces.append(trace)
    print(f"Q: {question[:74]}")
    for turn in trace:
        mark = "FAILED" if turn.failed else "ok"
        args = str(turn.arguments)[:52]
        print(f"   {mark:7} {turn.tool:18} {turn.seconds:5.2f}s  {args}")
        if turn.failed:
            print(f"           -> {turn.result[:60]}")
    print(f"   A: {answer[:260]}")
    print()

used = [[turn.tool for turn in trace] for trace in traces]
print("What the traces show:")
print(f"  question 1 used {', '.join(sorted(set(used[0])))}")
print(f"  question 2, asked to compute, called "
      f"{'arithmetic' if 'arithmetic' in used[1] else 'no arithmetic tool'}"
      f" and {used[1].count('query_warehouse')} warehouse quer"
      f"{'y' if used[1].count('query_warehouse') == 1 else 'ies'}")
print(f"  question 3, with no answer in the corpus, searched {used[2].count('search_documents')} "
      f"time{'s' if used[2].count('search_documents') != 1 else ''}")
