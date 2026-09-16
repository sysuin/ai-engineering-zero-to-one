# timeout: 1200
# The governed planner of 03_governed.py without its escape hatch: no `answerable` field, and
# no instruction to refuse. Everything else — metrics, dimensions, the SQL builder, the twenty
# questions and the scoring — is the same. Three runs, because the planner is a model.

import json
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

sys.path.insert(0, "code")
sys.path.insert(0, "code/15")
from _sqlset import QUESTIONS                                         # noqa: E402
from clarity.config import MODEL_FAST                                 # noqa: E402
from clarity.v0_7.warehouse import (DIMENSIONS, METRICS, DimensionName,  # noqa: E402
                                    Filter, MetricName, Warehouse)

DB = "data/meridian/warehouse/meridian.db"
RUNS = 3


class SpecWithoutHatch(BaseModel):
    metric: MetricName = Field(description="Which defined metric to compute.")
    group_by: list[DimensionName] = Field(
        default_factory=list, description="Dimensions to break the metric down by.")
    filters: list[Filter] = Field(
        default_factory=list, description="Restrictions the question states.")
    order: Literal["highest first", "lowest first", "none"] = "none"
    limit: int | None = Field(default=None, ge=1, le=100)


class NoHatch(Warehouse):
    def plan(self, question: str) -> SpecWithoutHatch | None:
        catalogue = "\n".join(
            f"  {name}: {m['label']} — {m['note']}" for name, m in METRICS.items())
        return self.client.chat.completions.parse(
            model=MODEL_FAST, temperature=0, max_completion_tokens=500,
            response_format=SpecWithoutHatch,
            messages=[{"role": "system", "content":
                       "Translate the question into a query specification.\n\n"
                       f"Metrics available:\n{catalogue}\n\n"
                       f"Dimensions available: {', '.join(DIMENSIONS)}\n\n"
                       "Use only these. Include a filter only for something the "
                       "question explicitly states."},
                      {"role": "user", "content": question}],
        ).choices[0].message.parsed

    def ask(self, question: str):
        spec = self.plan(question)
        if spec is None:
            return None
        sql, params = self.to_sql(spec)
        con = sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
        try:
            return spec, con.execute(sql, params).fetchall()
        finally:
            con.close()


def reference(sql: str):
    con = sqlite3.connect(DB)
    try:
        return con.execute(sql).fetchall()
    finally:
        con.close()


def same(a, b, tolerance: float = 0.01) -> bool:                       # as in 03_governed.py
    if a is None or b is None or len(a) != len(b):
        return False
    for row_a, row_b in zip(a, b):
        for vb in row_b:
            if isinstance(vb, (int, float)):
                if not any(isinstance(va, (int, float))
                           and abs(va - vb) <= tolerance * max(1.0, abs(vb))
                           for va in row_a):
                    return False
            elif not any(str(va).strip().lower() == str(vb).strip().lower() for va in row_a):
                return False
    return True


warehouse = NoHatch()


def attempt(pair):
    question, ref = pair
    out = warehouse.ask(question)
    if out is None:
        return question, None, False
    spec, rows = out
    return question, spec, same(rows, reference(ref))


REFUSED_BY_03 = ["How many customers are in the Midwest region?",
                 "Which supplier has the most products?",
                 "How many order lines had a discount of 15 percent?",
                 "Which region grew revenue most between 2023 and 2025?",
                 "How many suppliers are based outside the United States?"]
scores, forced = [], {}
for run in range(RUNS):
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(attempt, QUESTIONS))
    scores.append(sum(ok for _, _, ok in results))
    for question, spec, ok in results:
        if question in REFUSED_BY_03:
            forced.setdefault(question, []).append((spec, ok))

n = len(QUESTIONS)
print(f"{n} questions, {RUNS} runs, no way to say 'this cannot be expressed'\n")
print("  correct of all, each run   " + "   ".join(f"{s}/{n} ({s / n:.0%})" for s in scores))
naive = json.loads(Path("code/15/_naive.json").read_text())
governed = json.loads(Path("code/15/_governed.json").read_text())
print(f"  for comparison: free-form SQL {naive['correct']}/{n} in 01; with the escape hatch "
      f"{governed['correct']}/{n} in 03,")
print(f"  which answered {governed['answered']} and got "
      f"{governed['answered'] - governed['correct']} of those wrong\n")
print("  the five questions 03 refused, as this planner answered them on the first run:")
for question in REFUSED_BY_03:
    spec, ok = forced[question][0]
    print(f"    {question[:52]:<53} metric={spec.metric:<10} {'right' if ok else 'wrong'}")
print(f"\n  of those {len(REFUSED_BY_03) * RUNS} forced answers across all runs, right: "
      f"{sum(ok for runs in forced.values() for _, ok in runs)}")
growth, _ = forced["Which region grew revenue most between 2023 and 2025?"][0]
print("\n  how it answered 'which region grew revenue most between 2023 and 2025?':")
print(f"    metric={growth.metric}, group_by={growth.group_by}, order={growth.order!r}, "
      f"limit={growth.limit}")
print("    filters: " + ", ".join(f"{f.dimension} {f.operator} {f.value}" for f in growth.filters))
