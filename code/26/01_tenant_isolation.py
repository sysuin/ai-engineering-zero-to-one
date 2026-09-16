# timeout: 600
# The test that proves isolation, and the one-character change that breaks it.

import json
import sys
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

sys.path.insert(0, "code")
from clarity.v0_17.store import Document, Run, open_store       # noqa: E402

DB = Path("data/meridian/tenants.db")
DB.unlink(missing_ok=True)
engine = open_store(f"sqlite:///{DB}")

with Session(engine) as session:
    for tenant, questions in (("meridian", ["revenue in 2024 Q3", "margin in Q1"]),
                              ("halvard", ["our Q3 numbers"])):
        for question in questions:
            session.add(Run(tenant=tenant, trace_id="0" * 32, question=question,
                            answer="…"))
        session.add(Document(tenant=tenant, source=f"{tenant}-qbr.md",
                             kind="quarterly-review", sha256="a" * 64))
    session.commit()


def scoped(session, tenant: str):
    """The correct version: the filter is applied by the data layer, always."""
    return session.scalars(select(Run).where(Run.tenant == tenant)).all()


def unscoped(session, tenant: str):
    """
    The broken version, and note how ordinary it looks.

    Somebody adds an endpoint in a hurry, forgets the `where`, and every test that
    checks "does this return runs" still passes. The only test that fails is one
    written to ask a different question.
    """
    return session.scalars(select(Run)).all()


print("Two tenants in one database: meridian has 2 runs, halvard has 1.\n")
results = {}
with Session(engine) as session:
    for label, query in (("scoped query", scoped), ("missing WHERE", unscoped)):
        rows = query(session, "meridian")
        leaked = [r for r in rows if r.tenant != "meridian"]
        results[label] = {"returned": len(rows), "leaked": len(leaked)}
        verdict = "PASS" if not leaked else f"FAIL — {len(leaked)} row(s) leaked"
        print(f"  {label:<18}returned {len(rows)} rows to meridian   {verdict}")

json.dump(results, open("code/26/_tenants.json", "w"), indent=1)

print()
print("The isolation test is not 'does the query return data'. It is:")
print()
print("  ask as tenant A, assert that every row returned belongs to A")
print()
print("Write it once per table and run it on every build. It is three lines, it")
print("catches a missing WHERE clause, and a missing WHERE clause is one of the")
print("easiest ways for one customer to see another's data.")
print()
print("Two stronger versions, in increasing order of how much sleep they buy:")
print()
print("  a session variable    set the tenant on the connection and let the data")
print("                        layer apply it, so forgetting is not possible")
print("  row-level security    Postgres enforces the predicate itself, so a")
print("                        hand-written query cannot cross — unless it runs")
print("                        as a role the policy does not apply to")
print()
print("Both are more work than a WHERE clause and both remove a class of bug rather")
print("than an instance of it, which is the trade worth making for exactly this one.")
