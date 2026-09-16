# timeout: 600
# Every round of the loop sends the whole conversation again. What a question costs, round by
# round, when its calls arrive together and when they arrive one at a time.

import json
import sqlite3

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
DB = "file:data/meridian/warehouse/meridian.db?mode=ro"
TOOLS = [{"type": "function", "function": {
    "name": "monthly_revenue", "strict": True,
    "description": "Revenue for every month of one year, by region, as CSV rows.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["year"],
                   "properties": {"year": {"type": "integer"}}}}}]
QUESTION = ("Using monthly revenue by region for 2023, 2024 and 2025, which region's December "
            "revenue grew most from 2023 to 2025?")


def monthly_revenue(year: int) -> str:
    con = sqlite3.connect(DB, uri=True)
    rows = con.execute("SELECT strftime('%m', order_date), region, ROUND(SUM(revenue)) FROM v_sales "
                       "WHERE year = ? GROUP BY 1, 2 ORDER BY 1, 2", (year,)).fetchall()
    con.close()
    return "month,region,revenue\n" + "\n".join(f"{m},{r},{int(v)}" for m, r, v in rows)


def run(parallel: bool) -> list[dict]:
    messages = [{"role": "user", "content": QUESTION}]
    rounds = []
    for _ in range(6):
        reply = client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=800, tools=TOOLS,
            parallel_tool_calls=parallel, messages=messages)
        usage, message = reply.usage, reply.choices[0].message
        cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        rounds.append({"prompt": usage.prompt_tokens, "cached": cached,
                       "completion": usage.completion_tokens, "calls": len(message.tool_calls or [])})
        if not message.tool_calls:
            rounds[-1]["answer"] = (message.content or "").strip().replace("\n", " ")
            return rounds
        messages.append(message)
        for call in message.tool_calls:
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": monthly_revenue(**json.loads(call.function.arguments))})
    return rounds


con = sqlite3.connect(DB, uri=True)
december = dict(con.execute(
    "SELECT region, SUM(CASE WHEN year = 2025 THEN revenue ELSE -revenue END) FROM v_sales "
    "WHERE year IN (2023, 2025) AND strftime('%m', order_date) = '12' GROUP BY region").fetchall())
con.close()
print(f"the right answer: {max(december, key=december.get)}\n")
totals = {}
for parallel in (True, False):
    rounds = run(parallel)
    label = "calls together" if parallel else "one call per round"
    print(f"{label}")
    print(f"  {'round':>5}{'calls':>7}{'prompt':>9}{'cached':>9}{'output':>8}")
    for n, r in enumerate(rounds, start=1):
        print(f"  {n:>5}{r['calls']:>7}{r['prompt']:>9,}{r['cached']:>9,}{r['completion']:>8,}")
    totals[label] = sum(r["prompt"] for r in rounds)
    if len(rounds) > 2:
        growth = [b["prompt"] - a["prompt"] for a, b in zip(rounds, rounds[1:])]
        print(f"  each round's prompt grew by {', '.join(f'{g:,}' for g in growth)} tokens")
    print(f"  total prompt tokens {totals[label]:,} over {len(rounds)} model calls")
    print(f"  answer ends: …{rounds[-1].get('answer', '(ran out of rounds)')[-120:]}\n")

together, apart = totals["calls together"], totals["one call per round"]
print(f"one call per round sent {apart / together:.1f}x the prompt tokens of calls together")
