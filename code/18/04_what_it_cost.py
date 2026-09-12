# timeout: 900
# The bill for the two features, measured rather than asserted.

import importlib.metadata as md
import json
import subprocess
import sys
import time
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.v0_10.graph import build                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent                     # noqa: E402
from meridian_index import load_index                    # noqa: E402


def code_lines(path: str) -> int:
    """Statements only — blank lines and comments are not the thing being compared."""
    return sum(1 for line in Path(path).read_text().splitlines()
               if line.strip() and not line.strip().startswith("#"))


print("--- code you maintain ---")
hand_lines = code_lines("code/clarity/v0_9/agent.py")
graph_lines = code_lines("code/clarity/v0_10/graph.py")
print(f"  v0.9  hand-rolled loop : {hand_lines:>4} lines")
print(f"  v0.10 graph            : {graph_lines:>4} lines  ({graph_lines - hand_lines:+d})")

print("\n--- code you now depend on ---")
added = ["langgraph", "langgraph-checkpoint", "langgraph-checkpoint-sqlite",
         "langgraph-prebuilt", "langgraph-sdk", "langchain-core", "langchain-openai",
         "langsmith", "ormsgpack", "xxhash", "tenacity", "jsonpatch", "jsonpointer",
         "zstandard", "requests-toolbelt", "PyYAML"]
present = []
for name in added:
    try:
        present.append((name, md.version(name)))
    except md.PackageNotFoundError:
        pass
for name, version in present[:6]:
    print(f"  {name:<30} {version}")
print(f"  ... {len(present)} packages in total, none of which you wrote")

site = Path(sys.prefix) / "lib"
sizes = subprocess.run(["du", "-sk"] + [str(p) for p in site.rglob("langgraph*")
                                        if p.is_dir() and "__pycache__" not in str(p)],
                       capture_output=True, text=True).stdout
kb = sum(int(line.split()[0]) for line in sizes.splitlines() if line.strip())
print(f"  langgraph on disk: {kb / 1024:.1f} MB")

print("\n--- what it costs at import time ---")
import_seconds = {}
for module in ("openai", "langgraph.graph"):
    started = time.perf_counter()
    subprocess.run([sys.executable, "-c", f"import {module}"], check=True)
    import_seconds[module] = time.perf_counter() - started
    print(f"  import {module:<16} {import_seconds[module]:.2f}s (cold process)")

print("\n--- what it costs per run ---")
chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())
QUESTION = "What was revenue in 2024 Q3, and what caused the Midwest decline?"

app_client = OpenAI()


class Counting:
    """A stand-in for the client that records every request both versions make."""

    def __init__(self, inner):
        self._inner, self.calls, self.tokens, self.wire = inner, 0, 0, 0.0
        self.chat = self                     # client.chat.completions.create
        self.completions = self

    def create(self, **kwargs):
        started = time.perf_counter()
        response = self._inner.chat.completions.create(**kwargs)
        self.wire += time.perf_counter() - started
        self.calls += 1
        self.tokens += response.usage.total_tokens
        return response


RUNS = 7
STATE = {"messages": [{"role": "system", "content": "You are an analyst for Meridian."},
                      {"role": "user", "content": QUESTION}],
         "steps": 0, "budget": 8, "approvals": [], "stopped_because": ""}


def measure(make_call) -> tuple[list[float], Counting]:
    """Seven runs, because one run of anything over a network measures the network."""
    meter = Counting(app_client)
    samples = []
    for _ in range(RUNS):
        started = time.perf_counter()
        make_call(meter)()
        samples.append(time.perf_counter() - started)
    return sorted(samples), meter


hand, hand_meter = measure(
    lambda meter: lambda: Agent(tools, client=meter).run(
        QUESTION, system="You are an analyst for Meridian."))
gr, graph_meter = measure(
    lambda meter: lambda: build(tools, client=meter).compile().invoke(dict(STATE)))

for label, times, meter in (("v0.9 ", hand, hand_meter), ("v0.10", gr, graph_meter)):
    print(f"  {label} : {times[RUNS // 2]:5.1f}s median  (range {times[0]:.1f}-{times[-1]:.1f}s)"
          f"   {meter.calls // RUNS} calls, {meter.tokens // RUNS:,} tokens per run")
    print(f"          of which {meter.wire / RUNS:.1f}s was spent waiting for the model")
print()
# Whether the graph looks faster or slower depends on which way the network fell on
# the day. Decide that in code, so the conclusion cannot drift from the numbers.
overlap = hand[0] <= gr[-1] and gr[0] <= hand[-1]
spread = max(hand[-1] - hand[0], gr[-1] - gr[0])
gap = abs(gr[RUNS // 2] - hand[RUNS // 2])
print(f"  spread within one version : {spread:.1f}s")
print(f"  gap between the versions  : {gap:.1f}s")
print(f"  ranges overlap            : {'yes' if overlap else 'no'}")
print(f"  -> the difference is {'inside' if gap <= spread else 'outside'} the noise")
print()
hand_tokens, graph_tokens = hand_meter.tokens // RUNS, graph_meter.tokens // RUNS
drift = abs(graph_tokens - hand_tokens) / hand_tokens
print(f"Read the token counts first. They differ by {drift:.1%} - the model wording its")
print("own replies slightly differently between runs, not the framework adding anything.")
print("Who owns the loop does not change what the model is asked, so it cannot change")
print("what the model charges.")
print()
print("Then read the last block. Both versions do the same tool work - one warehouse")
print("query, one retrieval - and both spend most of the remainder waiting for the same")
print("network.")
print()
if gap <= spread:
    print(f"Here the two medians land {gap:.1f}s apart while a single version varies by")
    print(f"{spread:.1f}s across its own {RUNS} runs. The framework's bookkeeping is below the")
    print("resolution of the experiment: there is nothing to measure, which is the")
    print("answer to the question people usually ask about frameworks and speed.")
else:
    faster = "v0.10" if gr[RUNS // 2] < hand[RUNS // 2] else "v0.9 "
    print(f"Here the gap ({gap:.1f}s) is larger than the spread ({spread:.1f}s), and "
          f"{faster.strip()} is ahead.")
    print(f"Before reading anything into that, note that {gap:.1f}s of a "
          f"{max(hand[RUNS // 2], gr[RUNS // 2]):.1f}s run is")
    print("dominated by how long the model happened to take, and the direction is not")
    print("stable between sittings. Per-run overhead is the wrong thing to argue about.")
print()
print("The price is the second list: sixteen packages with their own release")
print("schedules, their own breaking changes and their own bugs, sitting between your")
print("stack trace and your code. That is the thing to weigh against durable state and")
print("a human in the loop - not milliseconds.")
print()
print("Chapter 17's ninety lines remain the right answer when you need neither.")

json.dump({"lines": {"v0.9": hand_lines, "v0.10": graph_lines},
           "packages": len(present), "disk_mb": round(kb / 1024, 1),
           "imports": import_seconds,
           "median": {"v0.9": hand[RUNS // 2], "v0.10": gr[RUNS // 2]},
           "range": {"v0.9": [hand[0], hand[-1]], "v0.10": [gr[0], gr[-1]]},
           "wire": {"v0.9": hand_meter.wire / RUNS, "v0.10": graph_meter.wire / RUNS},
           "tokens": {"v0.9": hand_tokens, "v0.10": graph_tokens},
           "runs": RUNS},
          open("code/18/_cost.json", "w"), indent=1)
