# What a tool returns is a prompt too, and it is paid for in tokens. The same
# rows in four formats, and what happens when a large result is cut to fit.

import csv
import io
import json
import sqlite3

import tiktoken

from clarity.config import MODEL_FAST

encoder = tiktoken.encoding_for_model(MODEL_FAST)
DB = "file:data/meridian/warehouse/meridian.db?mode=ro"
con = sqlite3.connect(DB, uri=True)


def fetch(sql: str) -> tuple[list[str], list[tuple]]:
    cursor = con.execute(sql)
    return [c[0] for c in cursor.description], cursor.fetchall()


def as_objects(columns, rows, **options) -> str:
    return json.dumps([dict(zip(columns, r)) for r in rows], **options)


columns, rows = fetch("""
    SELECT region, quarter, ROUND(SUM(revenue), 2) AS revenue,
           COUNT(DISTINCT order_id) AS orders
    FROM v_sales WHERE year = 2025 GROUP BY region, quarter""")

text = io.StringIO()
csv.writer(text, lineterminator="\n").writerows([columns, *rows])
formats = {
    "JSON, one object per row": as_objects(columns, rows, indent=2),
    "JSON, compact objects":    as_objects(columns, rows),
    "JSON, columns + rows":     json.dumps({"columns": columns, "rows": rows}),
    "CSV":                      text.getvalue(),
}
sizes = {label: len(encoder.encode(v)) for label, v in formats.items()}
print(f"{len(rows)} rows of 2025 revenue and orders by region and quarter\n")
for label, tokens in sizes.items():
    ratio = tokens / min(sizes.values())
    print(f"  {label:26} {tokens:>5} tokens   {ratio:.1f}x the smallest")

# A result too large to send, cut the way Clarity v0.8 cuts it: at 4,000
# characters of text.
columns, rows = fetch("""
    SELECT customer, quarter, ROUND(SUM(revenue), 2) AS revenue
    FROM v_sales WHERE year = 2025 GROUP BY customer, quarter""")
full = as_objects(columns, rows)
cut = full[:4000]
try:
    json.loads(cut)
    parsed = "parses"
except json.JSONDecodeError as error:
    parsed = f"does not parse ({error.msg})"
complete = cut.count("}")                   # no value contains a brace
partial = " and one partial row" if cut.count("{") > complete else ""
print(f"\n{len(rows)} rows, {len(full):,} characters as JSON; "
      "cut at 4,000 characters:")
print(f"  the text {parsed}")
print(f"  it holds {complete} complete rows{partial}")
print(f"  nothing in it says that {len(rows) - complete} rows are missing")

kept = rows[:40]
honest = json.dumps({
    "columns": columns, "rows": kept,
    "returned": len(kept), "total": len(rows),
    "note": f"First {len(kept)} of {len(rows)} rows. Add a filter or group "
            "by fewer dimensions to see the rest."})
print(f"\ncut by rows instead, and say so: {len(encoder.encode(honest))} "
      "tokens of valid JSON,\nand the model knows what it has not seen")
