# timeout: 900
# Execution-guided repair: run the query, and if the result is an error, empty or NULL, show
# the model what came back and let it try once more. Measured on all twenty questions.

import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code/15")
from _sqlset import QUESTIONS, SCHEMA        # noqa: E402
from clarity.config import MODEL_FAST        # noqa: E402

DB = "data/meridian/warehouse/meridian.db"
client = OpenAI()
SYSTEM = (f"You write SQLite queries. Schema:\n{SCHEMA}\n"
          "Reply with the query only, no explanation, no code fences.")


def ask(messages):
    reply = client.chat.completions.create(model=MODEL_FAST, temperature=0,
                                           max_completion_tokens=400, messages=messages)
    return re.sub(r"^```\w*\n?|```$", "", (reply.choices[0].message.content or "").strip(),
                  flags=re.M).strip()


def run(sql):
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return con.execute(sql).fetchall(), None
    except sqlite3.Error as error:
        return None, str(error)
    finally:
        con.close()


def suspicious(rows, error) -> str | None:
    if error:
        return f"The query failed: {error}"
    if not rows:
        return "The query returned no rows."
    if all(v is None for row in rows for v in row):
        return "The query returned NULL. A filter may be on the wrong column or use a value the column does not hold."
    return None


def same(a, b):
    if a is None or b is None or len(a) != len(b):
        return False
    for ra, rb in zip(a, b):
        for x, y in zip(ra, rb):
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                if abs(x - y) > 0.01 * max(1.0, abs(y)):
                    return False
            elif str(x).strip().lower() != str(y).strip().lower():
                return False
    return True


def solve(item):
    question, reference = item
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    sql = ask(messages)
    rows, error = run(sql)
    first = same(rows, run(reference)[0])
    problem = suspicious(rows, error)
    repaired = False
    if problem:
        messages += [{"role": "assistant", "content": sql},
                     {"role": "user", "content": problem + " Write a corrected query."}]
        rows, error = run(ask(messages))
        repaired = True
    return question, first, same(rows, run(reference)[0]), repaired


with ThreadPoolExecutor(max_workers=10) as pool:
    results = list(pool.map(solve, QUESTIONS))

before = sum(r[1] for r in results)
after = sum(r[2] for r in results)
retried = [r for r in results if r[3]]
print(f"right on the first attempt:   {before}/{len(results)}")
print(f"right after one repair round: {after}/{len(results)}")
print(f"queries that triggered a repair: {len(retried)}")
for question, first, final, _ in retried:
    print(f"  {'fixed' if final and not first else 'still wrong' if not final else 'was right'}: {question}")
wrong_but_plausible = [q for q, first, final, retried_ in results if not final and not retried_]
print(f"\nwrong answers that looked fine, so nothing triggered a repair: {len(wrong_but_plausible)}")
for question in wrong_but_plausible:
    print(f"  {question}")
