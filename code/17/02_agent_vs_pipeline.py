# timeout: 1800
# The question this chapter exists to answer: does this need an agent?
#
# Two task shapes, both done both ways, measured for accuracy, cost and time.

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from clarity.v0_9.agent import Agent, Budget             # noqa: E402
from meridian_index import load_index                    # noqa: E402

chunks, vectors = load_index()
retriever, warehouse = Retriever(chunks, vectors), Warehouse()
tools = build_tools(retriever, warehouse)
agent = Agent(tools, budget=Budget(steps=8, seconds=120))

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool. When you have enough, answer.")

# ---------------------------------------------------------------- task A: known shape
# Every quarter needs the same three things, in the same order, every time.
QUARTERS = [(2024, 3), (2024, 4), (2025, 1), (2025, 2)]


def pipeline_briefing(year: int, quarter: int) -> tuple[str, int, float]:
    """Three fixed calls. No decisions, because there are none to make."""
    started = time.time()
    tokens = 0

    revenue = warehouse.ask(f"revenue in {year} Q{quarter}")
    margin = warehouse.ask(f"margin_pct in {year} Q{quarter}")
    # v0.6 infers the filter from the question itself, so naming the quarter is
    # enough — Chapter 14's facet extraction doing the work the pipeline would
    # otherwise hard-code.
    passages = retriever.search(
        f"{year} Q{quarter} commentary and cause of the largest movement", k=4)

    from openai import OpenAI
    from clarity.config import MODEL_FAST
    context = "\n\n".join(f"[{p.source}] {p.text}" for p in passages)
    response = OpenAI().chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "system", "content":
                   "Write a three-sentence briefing from the figures and excerpts. "
                   "Use only what is given."},
                  {"role": "user", "content":
                   f"revenue={revenue.rows if revenue else None}\n"
                   f"margin_pct={margin.rows if margin else None}\n\n"
                   f"<excerpts>\n{context}\n</excerpts>"}])
    tokens += response.usage.total_tokens
    return ((response.choices[0].message.content or "").strip(), tokens,
            time.time() - started)


def agent_briefing(year: int, quarter: int):
    run = agent.run(f"Write a three-sentence briefing for {year} Q{quarter}: revenue, "
                    f"gross margin, and the cause of the largest movement.", SYSTEM)
    return run.answer, run.tokens, run.seconds, len(run.steps)


print("Task A — a briefing with a known shape, for four quarters\n")
rows_a = {"pipeline": [], "agent": []}
with ThreadPoolExecutor(max_workers=4) as pool:
    pipe = list(pool.map(lambda q: pipeline_briefing(*q), QUARTERS))
    ag = list(pool.map(lambda q: agent_briefing(*q), QUARTERS))

for (year, quarter), (pa, pt, ps), (aa, at, asec, steps) in zip(QUARTERS, pipe, ag):
    # Every briefing must contain the true revenue figure to count as correct.
    truth = warehouse.ask(f"revenue in {year} Q{quarter}").rows[0][0]
    stamp = f"{truth:,.0f}".replace(",", "")
    p_ok = stamp[:6] in pa.replace(",", "")
    a_ok = stamp[:6] in aa.replace(",", "")
    rows_a["pipeline"].append((p_ok, pt, ps, 3))
    rows_a["agent"].append((a_ok, at, asec, steps))
    print(f"  {year} Q{quarter}   pipeline {'ok ' if p_ok else 'WRONG'} "
          f"{pt:>6,}tok {ps:5.1f}s 3 calls    "
          f"agent {'ok ' if a_ok else 'WRONG'} {at:>6,}tok {asec:5.1f}s {steps} steps")


def summarise(rows):
    ok = sum(r[0] for r in rows)
    return ok / len(rows), sum(r[1] for r in rows) / len(rows), \
        sum(r[2] for r in rows) / len(rows), sum(r[3] for r in rows) / len(rows)


print(f"\n  {'':10} {'correct':>8} {'tokens':>9} {'seconds':>9} {'calls':>7}")
for name, rows in rows_a.items():
    acc, tok, sec, calls = summarise(rows)
    print(f"  {name:10} {acc:>7.0%} {tok:>9,.0f} {sec:>9.1f} {calls:>7.1f}")

# ---------------------------------------------------------------- task B: unknown shape
OPEN_QUESTIONS = [
    "Something changed in the Midwest during 2024. Work out what, and quantify it.",
    "Are customers complaining about a product problem? If so, which product, and "
    "does the sales data show an effect?",
    "Which supplier represents the biggest commercial risk, and why?",
]

print("\n\nTask B — questions with no known shape\n")
rows_b = []
for question in OPEN_QUESTIONS:
    run = agent.run(question, SYSTEM)
    rows_b.append({"q": question, "steps": len(run.steps), "tokens": run.tokens,
                   "tools": run.tools_used, "stopped": run.stopped_because})
    print(f"  {question[:64]}")
    print(f"     {len(run.steps)} steps · {run.tokens:,} tokens · {run.seconds:.1f}s "
          f"· {run.stopped_because}")
    print(f"     route: {' -> '.join(run.tools_used)}")
    print(f"     {run.answer[:150]}")
    print()

print("A pipeline for these would have to be written per question, which is another way")
print("of saying it would have to be written by someone who already knew the answer.")

Path("code/17/_comparison.json").write_text(json.dumps(
    {"task_a": {k: summarise(v) for k, v in rows_a.items()}, "task_b": rows_b},
    indent=2, default=str))
