# timeout: 1200
# Text to SQL in twenty lines. It works, which is the problem.

import json
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, "code/15")
from _sqlset import QUESTIONS, SCHEMA        # noqa: E402
from clarity.config import MODEL_FAST        # noqa: E402

client = OpenAI()
DB = "data/meridian/warehouse/meridian.db"


def generate(question: str) -> str:
    reply = (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "system", "content":
                   f"You write SQLite queries. Schema:\n{SCHEMA}\n"
                   "Reply with the query only, no explanation, no code fences."},
                  {"role": "user", "content": question}],
    ).choices[0].message.content or "").strip()
    return re.sub(r"^```\w*\n?|```$", "", reply, flags=re.M).strip()


def run(sql: str):
    con = sqlite3.connect(DB)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def same(a, b, tolerance: float = 0.01) -> bool:
    """Compare results, not query text. Two correct queries rarely look alike."""
    if a is None or b is None or len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        if len(row_a) != len(row_b):
            return False
        for x, y in zip(row_a, row_b):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if abs(x - y) > tolerance * max(1.0, abs(y)):
                    return False
            elif str(x).strip().lower() != str(y).strip().lower():
                return False
    return True


with ThreadPoolExecutor(max_workers=10) as pool:
    generated = list(pool.map(lambda qa: generate(qa[0]), QUESTIONS))

rows = []
executed = correct = 0
for (question, reference), sql in zip(QUESTIONS, generated):
    truth = run(reference)
    try:
        got = run(sql)
        executed += 1
    except Exception as error:
        rows.append({"q": question, "ok": False, "error": type(error).__name__})
        print(f"  ERROR   {question[:52]:54} {type(error).__name__}")
        continue
    ok = same(got, truth)
    correct += ok
    rows.append({"q": question, "ok": ok, "sql": sql})
    if not ok:
        print(f"  WRONG   {question[:52]:54} got {str(got)[:22]}  want {str(truth)[:22]}")

n = len(QUESTIONS)
print(f"\n{n} questions")
print(f"  ran without error   {executed}/{n}  {executed / n:.0%}")
print(f"  returned the right answer {correct}/{n}  {correct / n:.0%}")

Path("code/15/_naive.json").write_text(json.dumps(
    {"n": n, "executed": executed, "correct": correct, "rows": rows}, indent=2))

print()
print("Every query that ran, ran silently. A wrong number and a right number come back")
print("through the same code path, in the same shape, with no indication which is which.")
print()
print("This is the accuracy people demo, and it is measured here against a reference")
print("query rather than by reading the SQL and nodding — which is the only reason we")
print("know it is not 100%.")
