# timeout: 1200
# What a long run costs. Every round resends the whole conversation, so the prompt grows
# with each step and the tokens billed grow faster than the steps.

import sys
from types import SimpleNamespace

import tiktoken
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

encoder = tiktoken.encoding_for_model(MODEL_FAST)


class Recording:
    """A client that passes every call through and remembers what each one cost."""

    def __init__(self):
        self.real, self.calls = OpenAI(), []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        response = self.real.chat.completions.create(**request)
        results = [m["content"] for m in request["messages"]
                   if isinstance(m, dict) and m.get("role") == "tool"]
        self.calls.append((response.usage.prompt_tokens, results))
        return response


chunks, vectors = load_index()
recording = Recording()
agent = Agent(build_tools(Retriever(chunks, vectors), Warehouse()), client=recording,
              budget=Budget(steps=8, seconds=180))
run = agent.run("Which supplier represents the biggest commercial risk, and why? "
                "Check the documents and the sales data before you answer.",
                "You are an analyst for Meridian. Use tools for every fact.")

print(f"{len(run.steps)} steps, {len(recording.calls)} model calls, "
      f"stopped: {run.stopped_because}\n")


def tokens(texts: list[str]) -> int:
    return sum(len(encoder.encode(t)) for t in texts)


print(f"  {'call':>4} {'prompt tokens':>14} {'of which tool results':>22}")
for i, (prompt, results) in enumerate(recording.calls, 1):
    print(f"  {i:>4} {prompt:>14,} {tokens(results):>22,}")

billed = sum(p for p, _ in recording.calls)
last = recording.calls[-1][0]
from_results = sum(tokens(r) for _, r in recording.calls)
print(f"\ninput tokens billed across the run: {billed:,}")
print(f"the final prompt alone:             {last:,}")
print(f"tool results resent, in total:      {from_results:,} "
      f"({from_results / billed:.0%} of the input, estimated with the tokenizer)")

# The alternative: keep only the latest tool result in full, and each earlier one
# as a one-line note. Counted with the tokenizer, not billed, so it is an estimate.
compact = 0
for prompt, results in recording.calls:
    notes = [r[:120] for r in results[:-1]]
    compact += prompt - tokens(results) + tokens(results[-1:]) + tokens(notes)
print(f"\nwith older results reduced to one-line notes: about {compact:,} input tokens "
      f"({compact / billed:.0%} of what was billed)")
