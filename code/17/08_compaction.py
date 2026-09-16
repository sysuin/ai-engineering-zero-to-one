# timeout: 2400
# Compaction, done for real rather than estimated: older tool results are cut to a short note
# before each call. What it saves, and whether the agent still finds what it was looking for.

import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

NOTE = 160          # characters of an older result kept, after the tool's name


def compacted(messages: list) -> list:
    """Keep the latest round of tool results whole; reduce every earlier one to a note."""
    names = {}
    last_call = max((i for i, m in enumerate(messages)
                     if not isinstance(m, dict) and getattr(m, "tool_calls", None)), default=-1)
    out = []
    for i, m in enumerate(messages):
        if not isinstance(m, dict) and getattr(m, "tool_calls", None):
            names.update({c.id: c.function.name for c in m.tool_calls})
        if isinstance(m, dict) and m.get("role") == "tool" and i < last_call:
            name = names.get(m["tool_call_id"], "a tool")
            m = {**m, "content": f"[earlier {name} result, shortened] {m['content'][:NOTE]}"}
        out.append(m)
    return out


class Client:
    """Passes every call through — compacting first, if asked — and counts input tokens."""

    def __init__(self, compact: bool):
        self.real, self.compact, self.prompt_tokens = OpenAI(), compact, 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        if self.compact:
            request["messages"] = compacted(request["messages"])
        response = self.real.chat.completions.create(**request)
        self.prompt_tokens += response.usage.prompt_tokens
        return response


chunks, vectors = load_index()
TOOLS = build_tools(Retriever(chunks, vectors), Warehouse())
SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool. When you have enough, answer.")
# (question, strings a right answer must contain)
QUESTIONS = [
    ("Something changed in the Midwest during 2024. Work out what, and quantify it.", ["Halloway"]),
    ("Margin fell in 2025 Q1. Find out by how much, and what the company says caused it.", ["0.3", "Voss"]),
]
RUNS = 5


def one(question, must, compact):
    client = Client(compact)
    run = Agent(TOOLS, client=client, budget=Budget(steps=8, seconds=240)).run(question, SYSTEM)
    return client.prompt_tokens, len(run.steps), all(s.lower() in run.answer.lower() for s in must)


print(f"{len(QUESTIONS)} questions x {RUNS} runs; input tokens are the agent loop's own calls\n")
print(f"  {'question':30}{'memory':>18}{'right':>7}{'input tokens':>14}{'steps':>7}")
for question, must in QUESTIONS:
    for compact in (False, True):
        with ThreadPoolExecutor(max_workers=RUNS) as pool:
            rows = list(pool.map(lambda _: one(question, must, compact), range(RUNS)))
        label = "older results cut" if compact else "full transcript"
        print(f"  {question[:28]:30}{label:>18}{sum(r[2] for r in rows):>4}/{RUNS}"
              f"{statistics.median(r[0] for r in rows):>14,.0f}{statistics.median(r[1] for r in rows):>7.0f}")
print("\ninput tokens and steps are medians; 'right' means the answer names what the question needs")
