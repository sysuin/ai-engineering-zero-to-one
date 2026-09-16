# timeout: 900
# "A model asked for average margin by category, then overall, will average the averages."
# Eight ways of asking, sent to this chapter's twenty-line text-to-SQL program; each query is run
# and its overall figure compared with the three numbers it could be.

import json
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code/15")
from _sqlset import SCHEMA                   # noqa: E402
from clarity.config import MODEL_FAST        # noqa: E402

client = OpenAI()
db = sqlite3.connect("data/meridian/warehouse/meridian.db", check_same_thread=False)
QUESTIONS = [
    "What was the average gross margin by category in 2025, and overall?",
    "Show gross margin % for each product category in 2025, then the overall margin.",
    "Margin by category for 2025, plus a total row.",
    "For 2025, what was each category's gross margin percentage and the company-wide figure?",
    "Give me 2025 gross margin by category and for the business as a whole.",
    "What was average margin per category in 2025, and the overall average margin?",
    "List category margins for 2025 and the combined margin across all categories.",
    "2025 gross margin: by category, then overall.",
]


def generate(question: str) -> str:
    reply = (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=500,
        messages=[{"role": "system", "content":
                   f"You write SQLite queries. Schema:\n{SCHEMA}\n"
                   "Reply with the query only, no explanation, no code fences."},
                  {"role": "user", "content": question}],
    ).choices[0].message.content or "").strip()
    return re.sub(r"^```\w*\n?|```$", "", reply, flags=re.M).strip()


def one(sql: str) -> float:
    return db.execute(sql).fetchone()[0]


TRUE = one("SELECT 100.0 * SUM(gross_profit) / SUM(revenue) FROM v_sales WHERE year = 2025")
MEAN_OF_CATEGORIES = one("SELECT AVG(m) FROM (SELECT 100.0 * SUM(gross_profit) / SUM(revenue) AS m "
                         "FROM v_sales WHERE year = 2025 GROUP BY category)")
MEAN_OF_LINES = one("SELECT AVG(100.0 * gross_profit / revenue) FROM v_sales "
                    "WHERE year = 2025 AND revenue > 0")
CANDIDATES = {"total profit / total revenue": TRUE, "mean of the category margins": MEAN_OF_CATEGORIES,
              "mean of every line's margin": MEAN_OF_LINES}


def method(sql: str) -> str:
    """How the query computes a margin, read from its text rather than its result."""
    flat = re.sub(r"\s+", " ", sql.lower())
    if re.search(r"avg\([^)]*(margin|/)", flat):
        return "averages margins"
    if re.search(r"sum\(\s*gross_profit\s*\)[^;]*/[^;]*sum\(\s*revenue\s*\)", flat) or \
            re.search(r"gross_profit\s*/[^;]*revenue", flat):
        return "profit / revenue"
    return "other"


def ran(sql: str) -> str:
    try:
        db.execute(sql).fetchall()
        return "yes"
    except sqlite3.Error:
        return "no"


def overall(sql: str) -> str:
    """For a query that runs, which of the candidate figures its result contains."""
    numbers = [v for row in db.execute(sql).fetchall() for v in row if isinstance(v, (int, float))]
    for name, target in CANDIDATES.items():            # a ratio may come back as 0.31 or 31
        if any(abs(v - target) < 0.2 or abs(100 * v - target) < 0.2 for v in numbers):
            return {"total profit / total revenue": "profit / revenue"}.get(name, "averages margins")
    return "other"


print("the figure 'overall' could mean, for 2025:")
for name, value in CANDIDATES.items():
    print(f"  {name:<32}{value:6.2f}%")
with ThreadPoolExecutor(max_workers=8) as pool:
    queries = list(pool.map(generate, QUESTIONS))
rows = []
for question, sql in zip(QUESTIONS, queries):
    runs = ran(sql)
    rows.append((question, overall(sql) if runs == "yes" else method(sql), runs))
print(f"\n  {'asked':<48}{'margin computed as':<20}{'ran':>5}")
for question, how, runs in rows:
    print(f"  {question[:46]:<48}{how:<20}{runs:>5}")
print(f"\n  profit / revenue: {sum(r[1] == 'profit / revenue' for r in rows)} of {len(rows)};  "
      f"averaged margins: {sum(r[1] == 'averages margins' for r in rows)};  "
      f"ran in SQLite: {sum(r[2] == 'yes' for r in rows)}")
print("  (read from the result where the query ran, and from the SQL where it did not)")
json.dump([{"question": q, "sql": sql} for q, sql in zip(QUESTIONS, queries)],
          open("code/15/_average_of_averages.json", "w"), indent=2)
