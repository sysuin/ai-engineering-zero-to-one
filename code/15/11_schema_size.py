# timeout: 1200
# A real warehouse has more tables than the question needs, and some of them look relevant.
# The same twenty questions, with the schema the model sees padded by tables that do not help:
# unrelated ones, and ones whose names and columns look like the answer.

import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code/15")
from _sqlset import QUESTIONS, SCHEMA        # noqa: E402
from clarity.config import MODEL_FAST        # noqa: E402

client = OpenAI()
DB = "data/meridian/warehouse/meridian.db"

UNRELATED = """
hr_staff(staff_id, name, department, start_date, salary_band)
it_assets(asset_id, kind, assigned_to, purchased)
marketing_leads(lead_id, source, created, stage)
facilities_tickets(ticket_id, depot, opened, closed, category)
payroll_runs(run_id, period, total_gross, total_net)
audit_log(event_id, at, actor, action)
"""
LOOKALIKE = """
sales_archive(order_id, order_date, year, quarter, region, customer, revenue)   -- orders before 2023
revenue_forecast(year, quarter, region, revenue)                                 -- planning figures
sales_budget(year, quarter, region, category, revenue_target)
customer_revenue_summary(customer, year, revenue)                                -- refreshed monthly
supplier_scorecard(supplier, year, products_supplied, on_time_pct)
regional_kpis(region, year, quarter, orders, revenue, margin_pct)                 -- dashboard extract
"""
DECOYS = {line.split("(")[0].strip() for line in (UNRELATED + LOOKALIKE).splitlines() if "(" in line}


def generate(schema: str, question: str) -> str:
    reply = (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "system", "content":
                   f"You write SQLite queries. Schema:\n{schema}\n"
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


source = open("code/15/01_naive_text_to_sql.py").read()
same_src = source[source.index("def same"):source.index("with ThreadPoolExecutor")]
namespace: dict = {}
exec(same_src, namespace)                     # the same result comparison as 01
same = namespace["same"]

CONDITIONS = [("the schema as it is", SCHEMA),
              ("+ 6 unrelated tables", SCHEMA + UNRELATED),
              ("+ 6 look-alike tables", SCHEMA + LOOKALIKE),
              ("+ all 12", SCHEMA + UNRELATED + LOOKALIKE)]

print(f"{len(QUESTIONS)} questions; the decoy tables are only in the prompt, not in the database\n")
print(f"  {'schema shown':26}{'right':>8}{'ran':>6}{'used a decoy':>14}")
turns = []
for label, schema in CONDITIONS:
    with ThreadPoolExecutor(max_workers=10) as pool:
        generated = list(pool.map(lambda qa: generate(schema, qa[0]), QUESTIONS))
    right = ran = decoy = 0
    for (question, reference), sql in zip(QUESTIONS, generated):
        used = sorted(name for name in DECOYS if re.search(rf"\b{name}\b", sql))
        decoy += bool(used)
        turns += [(question, name) for name in used]
        try:
            got = run(sql)
        except sqlite3.Error:
            continue
        ran += 1
        right += same(got, run(reference))
    print(f"  {label:26}{right:>5}/{len(QUESTIONS)}{ran:>6}{decoy:>14}")

print("\nwhere the decoys were used, across the padded schemas")
for question, name in sorted(set(turns)):
    print(f"  {question[:52]:54}{name}")
