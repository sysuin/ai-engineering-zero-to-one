# timeout: 1800
# The question this chapter exists to answer: does this need an agent?
#
# Two task shapes, both done both ways, measured for accuracy, cost and time.

import json
import sqlite3
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

from openai import OpenAI                                 # noqa: E402

from clarity.config import MODEL_FAST                    # noqa: E402

chunks, vectors = load_index()
_shared = OpenAI()


class Counting:
    """
    An OpenAI client that adds up every token it is billed for.

    The first version of this listing counted the tokens each approach reported itself:
    the pipeline's final writing call and the agent's own loop. Both missed the model
    calls made *inside* the components — the warehouse's query planner, the retriever's
    facet extractor, the embedding of each search — and the pipeline, which makes three
    of those for every briefing, was undercounted most. Counting at the client is the
    only way to see every call.
    """

    def __init__(self) -> None:
        self.tokens = self.calls = 0
        outer = self

        class _Completions:
            def create(self, **kwargs):
                return outer._count(_shared.chat.completions.create(**kwargs))

            def parse(self, **kwargs):
                return outer._count(_shared.chat.completions.parse(**kwargs))

        class _Embeddings:
            def create(self, **kwargs):
                return outer._count(_shared.embeddings.create(**kwargs))

        self.chat = type("Chat", (), {"completions": _Completions()})()
        self.embeddings = _Embeddings()

    def _count(self, response):
        self.calls += 1
        self.tokens += getattr(getattr(response, "usage", None), "total_tokens", 0) or 0
        return response


def components(client: Counting):
    retriever = Retriever(chunks, vectors, client=client)
    warehouse = Warehouse(client=client)
    return retriever, warehouse

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. Never state a "
          "figure you did not obtain from a tool. When you have enough, answer.")

# ---------------------------------------------------------------- task A: known shape
# Every quarter needs the same three things, in the same order, every time.
QUARTERS = [(2024, 3), (2024, 4), (2025, 1), (2025, 2)]
RUNS = 2                                     # each quarter, each way, twice


def pipeline_briefing(year: int, quarter: int) -> tuple[str, int, float, int]:
    """Three fixed steps and a writing call. No decisions, because there are none to make."""
    started = time.time()
    client = Counting()
    retriever, warehouse = components(client)

    revenue = warehouse.ask(f"revenue in {year} Q{quarter}")
    margin = warehouse.ask(f"margin_pct in {year} Q{quarter}")
    # v0.6 infers the filter from the question itself, so naming the quarter is
    # enough — Chapter 14's facet extraction doing the work the pipeline would
    # otherwise hard-code.
    passages = retriever.search(
        f"{year} Q{quarter} commentary and cause of the largest movement", k=4)

    context = "\n\n".join(f"[{p.source}] {p.text}" for p in passages)
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "system", "content":
                   "Write a three-sentence briefing from the figures and excerpts. "
                   "Use only what is given."},
                  {"role": "user", "content":
                   f"revenue={revenue.rows if revenue else None}\n"
                   f"margin_pct={margin.rows if margin else None}\n\n"
                   f"<excerpts>\n{context}\n</excerpts>"}])
    return ((response.choices[0].message.content or "").strip(), client.tokens,
            time.time() - started, client.calls)


def new_agent(client: Counting) -> Agent:
    retriever, warehouse = components(client)
    return Agent(build_tools(retriever, warehouse), client=client,
                 budget=Budget(steps=8, seconds=120))


def agent_briefing(year: int, quarter: int):
    client = Counting()
    run = new_agent(client).run(
        f"Write a three-sentence briefing for {year} Q{quarter}: revenue, "
        f"gross margin, and the cause of the largest movement.", SYSTEM)
    return run.answer, client.tokens, run.seconds, len(run.steps), client.calls


def correct(text: str, year: int, quarter: int) -> bool:
    """A briefing counts only if it states both the true revenue and the true margin."""
    con = sqlite3.connect(f"file:{Warehouse(client=_shared).path}?mode=ro", uri=True)
    revenue, margin = con.execute(
        "SELECT SUM(revenue), ROUND(100.0 * SUM(gross_profit) / SUM(revenue), 1) "
        "FROM v_sales WHERE year = ? AND quarter = ?", (year, quarter)).fetchone()
    digits = text.replace(",", "")
    stated = f"{revenue:.0f}"[:4] in digits or f"{revenue / 1e6:.2f}" in digits
    return stated and f"{margin}" in digits


print(f"Task A — a briefing with a known shape, for four quarters, {RUNS} runs each\n")
rows_a = {"pipeline": [], "agent": []}
jobs = [q for q in QUARTERS for _ in range(RUNS)]
with ThreadPoolExecutor(max_workers=4) as pool:
    pipe = list(pool.map(lambda q: pipeline_briefing(*q), jobs))
    ag = list(pool.map(lambda q: agent_briefing(*q), jobs))

for (year, quarter), (pa, pt, ps, pc), (aa, at, asec, steps, ac) in zip(jobs, pipe, ag):
    p_ok = correct(pa, year, quarter)
    a_ok = correct(aa, year, quarter)
    rows_a["pipeline"].append((p_ok, pt, ps, pc))
    rows_a["agent"].append((a_ok, at, asec, ac))
    print(f"  {year} Q{quarter}  pipeline {'ok ' if p_ok else 'WRONG'} "
          f"{pt:>6,}tok {ps:5.1f}s {pc:>2} calls   "
          f"agent {'ok ' if a_ok else 'WRONG'} {at:>6,}tok {asec:5.1f}s {ac:>2} calls "
          f"{steps} steps")


def summarise(rows):
    ok = sum(r[0] for r in rows)
    return ok / len(rows), sum(r[1] for r in rows) / len(rows), \
        sum(r[2] for r in rows) / len(rows), sum(r[3] for r in rows) / len(rows)


print(f"\n  {'':10} {'correct':>8} {'tokens':>9} {'seconds':>9} {'calls':>7}")
for name, rows in rows_a.items():
    acc, tok, sec, calls = summarise(rows)
    print(f"  {name:10} {acc:>7.0%} {tok:>9,.0f} {sec:>9.1f} {calls:>7.1f}")

(p_acc, p_tok, p_sec, _), (a_acc, a_tok, a_sec, _) = map(summarise, rows_a.values())
agent_steps = [a[3] for a in ag]
speed = ("about as fast" if abs(a_sec - p_sec) < 0.1 * p_sec
         else "slower" if a_sec > p_sec else "faster")
print(f"\n  counting every model call, embeddings included, the agent used "
      f"{a_tok / p_tok:.1f}x the pipeline's tokens,\n  took "
      f"{min(agent_steps)} to {max(agent_steps)} steps for the same job, and was "
      f"{speed} on average ({a_sec:.1f}s against {p_sec:.1f}s)")

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
    client = Counting()
    run = new_agent(client).run(question, SYSTEM)
    rows_b.append({"q": question, "steps": len(run.steps), "tokens": client.tokens,
                   "tools": run.tools_used, "stopped": run.stopped_because})
    print(f"  {question[:64]}")
    print(f"     {len(run.steps)} steps · {client.tokens:,} tokens · {run.seconds:.1f}s "
          f"· {run.stopped_because}")
    print(f"     route: {' -> '.join(run.tools_used)}")
    print(f"     {run.answer[:150]}")
    print()

routes = {" -> ".join(r["tools"]) for r in rows_b}
exhausted = sum("exhausted" in r["stopped"] for r in rows_b)
print(f"{len(routes)} different routes for {len(rows_b)} questions; "
      f"{exhausted} of {len(rows_b)} runs hit a budget.\n")
print("A pipeline for these would have to be written per question, which is another way")
print("of saying it would have to be written by someone who already knew the answer.")

Path("code/17/_comparison.json").write_text(json.dumps(
    {"task_a": {k: summarise(v) for k, v in rows_a.items()}, "task_b": rows_b},
    indent=2, default=str))
