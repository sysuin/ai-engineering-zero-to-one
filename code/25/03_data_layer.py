# timeout: 600
# Five tables, and the question each one exists to answer.

import json
import sys
from pathlib import Path

from sqlalchemy import Integer, cast, func, select
from sqlalchemy.orm import Session

sys.path.insert(0, "code")
from clarity.v0_17.store import (Document, Evaluation, Feedback,   # noqa: E402
                                 Run, Step, open_store)

DB = Path("data/meridian/clarity.db")
DB.unlink(missing_ok=True)
engine = open_store()

with Session(engine) as session:
    runs = []
    for n, (question, tokens, cost, refused) in enumerate([
            ("What was revenue in 2024 Q3?", 1_650, 0.00033, False),
            ("Why did margin fall in 2025 Q1?", 2_410, 0.00051, False),
            ("What is Meridian's headcount?", 980, 0.00019, True),
            ("What was revenue in 2024 Q3?", 1_610, 0.00032, False)]):
        run = Run(tenant="meridian", trace_id=f"{n:032x}", question=question,
                  answer="…", tokens=tokens, cost_usd=cost, seconds=6.4,
                  refused=refused)
        run.steps = [Step(n=1, tool="query_warehouse", arguments={"question": question},
                          result_chars=180, seconds=2.9),
                     Step(n=2, tool="search_documents", arguments={"query": question},
                          result_chars=2_100, seconds=1.4)]
        runs.append(run)
        session.add(run)
    session.add(Feedback(run=runs[1], verdict="down", note="wrong quarter"))
    session.add(Evaluation(suite_version=1, layer="deterministic", score=0.92,
                           passed=True, git_sha="f8427cc"))
    session.add(Document(tenant="meridian", source="qbr-2024-Q3.md",
                         kind="quarterly-review", sha256="a" * 64))
    session.commit()

print("Five questions somebody will ask, and the query that answers each.\n")

with Session(engine) as session:
    queries = [
        ("what did we spend today, per tenant",
         select(Run.tenant, func.count(Run.id), func.sum(Run.cost_usd))
         .group_by(Run.tenant)),
        ("how often do we refuse",
         # SQLite has no boolean type, so summing one gives a boolean back.
         # Cast it, or the dashboard reports "True" refusals out of four.
         select(func.sum(cast(Run.refused, Integer)), func.count(Run.id))),
        ("which questions get asked twice",
         select(Run.question, func.count(Run.id).label("n"))
         .group_by(Run.question).having(func.count(Run.id) > 1)),
        ("what did the run the user complained about do",
         select(Run.question, Step.tool, Step.seconds)
         .join(Step, Step.run_id == Run.id)
         .join(Feedback, Feedback.run_id == Run.id)
         .where(Feedback.verdict == "down")),
        ("did the suite pass on the commit we shipped",
         select(Evaluation.layer, Evaluation.score, Evaluation.passed)
         .where(Evaluation.git_sha == "f8427cc")),
    ]
    summary = {}
    for label, query in queries:
        rows = session.execute(query).all()
        summary[label] = [list(map(str, r)) for r in rows]
        print(f"  {label}")
        for row in rows:
            print(f"      {'  '.join(str(v) for v in row)}")
        print()

json.dump(summary, open("code/25/_store.json", "w"), indent=1)
tables = list(__import__("sqlalchemy").inspect(engine).get_table_names())
print(f"  tables: {', '.join(sorted(tables))}")
print(f"  file:   {DB} ({DB.stat().st_size / 1024:.0f} KB)")

print()
print("None of those five questions can be answered from a log file, and all five")
print("get asked. That is the whole argument for a database here — not scale, which")
print("Clarity does not have, but *joins*: a complaint has to reach the run, the run")
print("has to reach its steps, and the deployment has to reach the eval that passed")
print("before it went out.")
print()
print("Three decisions in that schema are worth arguing about now rather than later.")
print()
print("Every table carries a tenant. Chapter 26 enforces it; the schema only has to")
print("make it possible, and retrofitting a tenant column onto a live system is a")
print("migration nobody enjoys twice.")
print()
print("Steps store arguments and result *sizes*, not results. §23.5's reasoning about")
print("what belongs in a trace applies to a database you own as much as to a vendor")
print("you do not — with one difference: here you can afford a separate table with")
print("its own retention if you decide you need the text.")
print()
print("And the run carries the trace id. Without it the database and the tracing")
print("backend are two accounts of the same afternoon that cannot be joined, which is")
print("the single most common gap between a system that has observability and a")
print("system that has dashboards.")
