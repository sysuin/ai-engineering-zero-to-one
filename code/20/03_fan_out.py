# timeout: 1800
# Six independent questions, done one after another and then all at once.

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

QUARTERS = ["2024 Q1", "2024 Q2", "2024 Q3", "2024 Q4", "2025 Q1", "2025 Q2"]
SYSTEM = ("You are an analyst for Meridian. Answer in one sentence: the revenue "
          "figure, and one thing the documents say about the quarter.")

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())


def brief(quarter: str) -> tuple[str, int]:
    run = Agent(tools, budget=Budget(steps=4)).run(
        f"What was revenue in {quarter}, and what does the quarterly review say "
        f"about it?", system=SYSTEM)
    return run.answer, run.tokens


def timed(fn) -> tuple[float, int]:
    started = time.perf_counter()
    answers = fn()
    return time.perf_counter() - started, sum(t for _, t in answers)


sequential, sequential_tokens = timed(lambda: [brief(q) for q in QUARTERS])
print(f"one after another : {sequential:5.1f}s   {sequential_tokens:,} tokens")

with ThreadPoolExecutor(max_workers=len(QUARTERS)) as pool:
    parallel, parallel_tokens = timed(lambda: list(pool.map(brief, QUARTERS)))
print(f"all at once       : {parallel:5.1f}s   {parallel_tokens:,} tokens")

speedup = sequential / parallel
print()
print(f"{speedup:.1f}x faster on the wall clock, "
      f"{parallel_tokens / sequential_tokens:.2f}x the tokens")
json.dump({"sequential": sequential, "parallel": parallel,
           "sequential_tokens": sequential_tokens,
           "parallel_tokens": parallel_tokens, "n": len(QUARTERS)},
          open("code/20/_fanout.json", "w"), indent=1)

print()
print("This is the multi-agent win, and it is worth being precise about what kind of")
print("win it is. Nothing got cheaper. The same six briefings cost the same tokens")
print("and made the same model calls; they simply waited at the same time instead of")
print("in a queue.")
print()
efficiency = speedup / len(QUARTERS)
if efficiency > 0.9:
    print(f"{len(QUARTERS)} workers gave {speedup:.1f}x, which is very nearly linear, "
          f"and it is linear")
    print("for an unglamorous reason: almost the whole of each run is spent waiting")
    print("for a model to reply, and waiting parallelises perfectly. Do not expect")
    print("this on work your own machine actually does.")
else:
    print(f"{len(QUARTERS)} workers gave {speedup:.1f}x rather than "
          f"{len(QUARTERS)}x. The missing part is the")
    print("work your own machine does — retrieval, the warehouse, JSON — which does")
    print("not overlap with itself as neatly as waiting on a network does.")
print()
print("The condition for this pattern is strict and easy to violate: the six tasks")
print("must not need each other. The moment quarter two's briefing depends on what")
print("quarter one found, you are back in a queue with extra machinery.")
