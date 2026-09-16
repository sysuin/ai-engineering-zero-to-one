# timeout: 900
# Schema linking with values: before generating SQL, find which words in the question are
# values in the data, and tell the model which column each one belongs to.

import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

sys.path.insert(0, "code/15")
from _sqlset import QUESTIONS, SCHEMA        # noqa: E402
from clarity.config import MODEL_FAST        # noqa: E402

DB = "data/meridian/warehouse/meridian.db"
con = sqlite3.connect(f"file:{DB}?mode=ro", uri=True)
client = OpenAI()

# Every distinct value of every short text column, mapped back to where it lives.
TEXT_COLUMNS = {"v_sales": ["region", "segment", "category", "channel", "supplier", "customer"],
                "suppliers": ["country", "tier"]}
VALUES: dict[str, list[str]] = {}          # the value exactly as stored -> where it lives
for table, columns in TEXT_COLUMNS.items():
    for column in columns:
        for (value,) in con.execute(f"SELECT DISTINCT {column} FROM {table}"):
            VALUES.setdefault(str(value), []).append(f"{table}.{column}")
print(f"value index: {len(VALUES):,} distinct values from {sum(map(len, TEXT_COLUMNS.values()))} columns\n")


def linked(question: str) -> list[str]:
    # Match without regard to case, but hand the model the value exactly as stored: SQLite's
    # = is case-sensitive, and a first version of this listing that lowercased the hint
    # produced WHERE supplier = 'voss industrial', which matches nothing.
    return [f"'{value}' is a value of {', '.join(where)}"
            for value, where in sorted(VALUES.items(), key=lambda kv: -len(kv[0]))
            if len(value) > 3 and re.search(rf"\b{re.escape(value)}\b", question, re.IGNORECASE)]


def generate(question: str, hints: list[str] | None) -> str:
    extra = ""
    if hints is not None:                                   # None means: no linking at all
        extra = ("\nValues found in the question:\n" + "\n".join(hints)) if hints else ""
        codes = ", ".join(f"'{v}'" for v, where in VALUES.items() if "suppliers.country" in where)
        extra += f"\nsuppliers.country holds only these values: {codes}."
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "system", "content": f"You write SQLite queries. Schema:\n{SCHEMA}{extra}\n"
                   "Reply with the query only, no explanation, no code fences."},
                  {"role": "user", "content": question}]).choices[0].message.content or ""
    return re.sub(r"^```\w*\n?|```$", "", reply.strip(), flags=re.M).strip()


def result(sql):
    try:
        return con.execute(sql).fetchall()
    except sqlite3.Error as error:
        return f"error: {error}"


TARGETS = [q for q in QUESTIONS if "Voss" in q[0] or "outside the United States" in q[0]]
for question, reference in TARGETS:
    hints = linked(question)
    with ThreadPoolExecutor(max_workers=2) as pool:
        plain, hinted = pool.map(lambda h: generate(question, h), (None, hints))
    print(question)
    print(f"  linked:   {hints or 'nothing'}")
    print(f"  truth     {result(reference)}")
    print(f"  no hints  {result(plain)}")
    print(f"  hints     {result(hinted)}\n")
