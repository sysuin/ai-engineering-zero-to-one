# timeout: 2400
# Questions that need several independent figures. Does the model ask for them in one reply, and
# what do running those calls at once, one after another, or forbidding parallel calls do to the
# rounds, the time and the answer?

import json
import re
import sqlite3
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                    # noqa: E402
from clarity.v0_6.retrieve import Retriever              # noqa: E402
from clarity.v0_7.warehouse import Warehouse             # noqa: E402
from clarity.v0_8.tools import build_tools               # noqa: E402
from meridian_index import load_index                    # noqa: E402

client = OpenAI()
chunks, vectors = load_index()
TOOLS = {t.name: t for t in build_tools(Retriever(chunks, vectors), Warehouse())}
DB = sqlite3.connect("data/meridian/warehouse/meridian.db", check_same_thread=False)
SYSTEM = "You are an analyst for Meridian. Use tools for every figure, then answer briefly."
RUNS = 3


def truth(sql: str) -> list[float]:
    return [row[0] for row in DB.execute(sql).fetchall()]


QUESTIONS = [
    ("What was revenue in 2023 Q4, 2024 Q4 and 2025 Q4?",
     truth("SELECT SUM(revenue) FROM v_sales WHERE quarter=4 AND year IN (2023,2024,2025) GROUP BY year")),
    ("What was gross margin, as a percentage, in 2024 Q1, 2024 Q2 and 2024 Q3?",
     truth("SELECT 100.0*SUM(gross_profit)/SUM(revenue) FROM v_sales WHERE year=2024 AND quarter<=3 GROUP BY quarter")),
    ("How many orders were there in each quarter of 2023?",
     truth("SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE year=2023 GROUP BY quarter")),
    ("What was 2024 revenue in the Midwest, the West and the Southwest?",
     truth("SELECT SUM(revenue) FROM v_sales WHERE year=2024 AND region IN ('Midwest','West','Southwest') GROUP BY region")),
]


def numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)]


def right(answer: str, wanted: list[float]) -> bool:
    found = numbers(answer)
    return all(any(abs(f - w) <= 0.006 * abs(w) or abs(f - w) < 0.06 for f in found) for w in wanted)


def run(question: str, arm: str) -> tuple[bool, int, int, float]:
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    rounds = calls = 0
    started = time.perf_counter()
    for _ in range(8):
        extra = {"parallel_tool_calls": False} if arm == "one call per reply" else {}
        reply = client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=700, messages=messages,
            tools=[t.schema() for t in TOOLS.values()], **extra).choices[0].message
        rounds += 1
        if not reply.tool_calls:
            return right(reply.content or "", dict(QUESTIONS)[question]), rounds, calls, \
                time.perf_counter() - started
        messages.append(reply)
        calls += len(reply.tool_calls)

        def execute(call):
            try:
                return json.dumps(TOOLS[call.function.name].run(**json.loads(call.function.arguments)),
                                  default=str)[:2000]
            except Exception as error:                          # noqa: BLE001
                return f"error: {error}"

        if arm == "several at once":
            with ThreadPoolExecutor(max_workers=8) as pool:
                results = list(pool.map(execute, reply.tool_calls))
        else:
            results = [execute(call) for call in reply.tool_calls]
        # paired by id, never by completion order
        messages += [{"role": "tool", "tool_call_id": c.id, "content": r}
                     for c, r in zip(reply.tool_calls, results)]
    return False, rounds, calls, time.perf_counter() - started


ARMS = ["several at once", "several, one after another", "one call per reply"]
print(f"{len(QUESTIONS)} questions needing 3 or 4 independent figures, {RUNS} runs each\n")
print(f"  {'tool calls in a reply':28}{'right':>8}{'model calls':>13}{'tool calls':>12}{'seconds':>9}")
for arm in ARMS:
    jobs = [q for q, _ in QUESTIONS for _ in range(RUNS)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(lambda q: run(q, arm), jobs))
    print(f"  {arm:28}{sum(r[0] for r in rows):>5}/{len(rows)}"
          f"{statistics.median(r[1] for r in rows):>13.0f}{statistics.median(r[2] for r in rows):>12.0f}"
          f"{statistics.median(r[3] for r in rows):>9.1f}")
print("\nmodel calls, tool calls and seconds are medians per question")
