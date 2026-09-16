# timeout: 3600
# "Reflect inside the loop can work, because the critic can call a tool." Tested: each draft an
# agent writes is reviewed twice — by a critic that can only reread it, and by one that can query
# the warehouse and search the documents — and all three versions are checked.

import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

chunks, vectors = load_index()
TOOLS = build_tools(Retriever(chunks, vectors), Warehouse())
client = OpenAI()
SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool. When you have enough, answer.")
CHECK = ("You are reviewing a colleague's draft answer before it is sent. Check every figure and "
         "claim against the tools, including whether the comparison the question implies was made "
         "with the right period. Then write the final answer: the draft if it holds, corrected if not.")
QUESTIONS = [
    ("Margin fell in 2025 Q1. Find out by how much, and what the company says caused it.", ["0.3", "Voss"]),
    ("Something changed in the Midwest during 2024. Work out what, and quantify it.", ["Halloway"]),
]
RUNS = 12


def right(answer: str, must: list[str]) -> bool:
    return all(s.lower() in answer.lower() for s in must)


def reread(question: str, draft: str) -> str:
    """A critic with nothing but the draft: the Chapter 9 kind."""
    return client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=700,
        messages=[{"role": "system", "content": "You are reviewing a colleague's draft answer before it "
                   "is sent. Check it carefully and write the final answer: the draft if it holds, "
                   "corrected if not."},
                  {"role": "user", "content": f"Question: {question}\n\nDraft:\n{draft}"}],
    ).choices[0].message.content or ""


def with_tools(question: str, draft: str) -> str:
    run = Agent(TOOLS, budget=Budget(steps=5, seconds=180)).run(
        f"Question: {question}\n\nDraft answer to check:\n{draft}", CHECK)
    return run.answer


def one(question: str, must: list[str]) -> tuple[bool, bool, bool]:
    draft = Agent(TOOLS, budget=Budget(steps=8, seconds=240)).run(question, SYSTEM).answer
    return right(draft, must), right(reread(question, draft), must), right(with_tools(question, draft), must)


print(f"{len(QUESTIONS)} questions x {RUNS} drafts; each draft reviewed by both critics\n")
print(f"  {'question':30}{'draft':>7}{'reread':>9}{'with tools':>12}")
changes = []
for question, must in QUESTIONS:
    with ThreadPoolExecutor(max_workers=RUNS) as pool:
        rows = list(pool.map(lambda _: one(question, must), range(RUNS)))
    for critic, k in (("reread", 1), ("with tools", 2)):
        changes.append((question[:28], critic, sum(1 for r in rows if r[k] and not r[0]),
                        sum(1 for r in rows if r[0] and not r[k])))
    print(f"  {question[:28]:30}{sum(r[0] for r in rows):>4}/{RUNS}{sum(r[1] for r in rows):>6}/{RUNS}"
          f"{sum(r[2] for r in rows):>9}/{RUNS}")
print("\n  drafts each critic fixed and broke")
for question, critic, fixed, broke in changes:
    print(f"  {question:30}{critic:>12}   fixed {fixed}, broke {broke}")
print("\nright: the answer names what the question needs (the fall and its cause; the lost account)")
