# timeout: 1200
# Five samples per question, each run and reduced to its answer. Does the majority fix the silent errors, and does
# disagreement among samples flag them — without knowing the right answer?

import sqlite3
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code/15")
from _sqlset import QUESTIONS, SCHEMA        # noqa: E402
from clarity.config import MODEL_FAST        # noqa: E402

client = OpenAI()
DB = "data/meridian/warehouse/meridian.db"
SAMPLES = 5


def generate(question: str) -> str:
    reply = (client.chat.completions.create(
        model=MODEL_FAST, temperature=1.0, max_completion_tokens=400,
        messages=[{"role": "system", "content":
                   f"You write SQLite queries. Schema:\n{SCHEMA}\n"
                   "Reply with the query only, no explanation, no code fences."},
                  {"role": "user", "content": question}],
    ).choices[0].message.content or "").strip()
    return reply.removeprefix("```sql").removeprefix("```").removesuffix("```").strip()


def execute(sql: str):
    """The answer a query gives, as a comparable value: the first column of each row, rounded.
    A sample that adds a column of revenue beside the region it names gives the same answer."""
    con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
    try:
        return tuple(round(row[0], 1) if isinstance(row[0], float) else row[0]
                     for row in con.execute(sql).fetchall())
    except Exception as error:                        # noqa: BLE001
        return f"error: {type(error).__name__}"
    finally:
        con.close()


def right(answer, reference: str) -> bool:
    return answer == execute(reference)


jobs = [q for q, _ in QUESTIONS for _ in range(SAMPLES)]
with ThreadPoolExecutor(max_workers=10) as pool:
    sqls = list(pool.map(generate, jobs))
results = [execute(s) for s in sqls]

rows = []
for n, (question, reference) in enumerate(QUESTIONS):
    mine = results[n * SAMPLES:(n + 1) * SAMPLES]
    modal, count = Counter(mine).most_common(1)[0]
    rows.append({"question": question, "first": right(mine[0], reference),
                 "majority": right(modal, reference), "agreement": count / SAMPLES,
                 "any": any(right(r, reference) for r in mine)})

print(f"{len(QUESTIONS)} questions, {SAMPLES} samples each at temperature 1.0\n")
print(f"  right on the first sample      {sum(r['first'] for r in rows)}/20")
print(f"  right by majority of results   {sum(r['majority'] for r in rows)}/20")
print(f"  right in at least one sample   {sum(r['any'] for r in rows)}/20")

unanimous = [r for r in rows if r["agreement"] == 1.0]
split = [r for r in rows if r["agreement"] < 1.0]
print(f"\n  all {SAMPLES} samples agreed:   {len(unanimous):>2} questions, "
      f"{sum(not r['majority'] for r in unanimous)} of them wrong")
print(f"  samples disagreed:      {len(split):>2} questions, "
      f"{sum(not r['majority'] for r in split)} of them wrong")

print("\nquestions the majority got wrong, with how many samples agreed on that answer:")
for r in rows:
    if not r["majority"]:
        print(f"  {r['agreement']:.0%}  {r['question']}")
