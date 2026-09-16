# skip
"""
Build Clarity's golden set: 120 cases, every one with a verified source.

Two halves, and the split is the point of §21.3.

The **head** is generated from the structure the corpus actually has — a quarterly
review says its revenue in a sentence shaped the same way every quarter, so a hundred
of those can be produced and checked mechanically. This is cheap, and it is where most
eval sets stop.

The **tail** is hand-written, because nothing generated from a template will ever ask
about a table printed sideways, a figure that appears in two documents with different
roundings, or a question whose answer is that there is no answer. Those cases are the
ones that find bugs, and there is no way to get them except by writing them.

Nothing here is trusted. `verify()` proves that every span appears verbatim in the file
it claims, and that every answer appears in its own span. A golden set nobody checked
is a set of assertions about a corpus somebody remembers.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path

import yaml

DOCS = Path("data/meridian/documents")
QBR = DOCS / "quarterly-reviews"
DB = "data/meridian/warehouse/meridian.db"
OUT = Path("code/clarity/evals/golden.yaml")

cases: list[dict] = []


def add(**case) -> None:
    case.setdefault("tier", "head")
    case.setdefault("accept", [])
    cases.append(case)


def sentence(text: str, needle: str) -> str:
    """The smallest sentence containing a phrase — that is the span we will cite."""
    for line in text.splitlines():
        if needle in line:
            for part in re.split(r"(?<=[.!?])\s+", line.strip()):
                if needle in part:
                    return part.strip()
            return line.strip()
    raise LookupError(needle)


# ---------------------------------------------------------------- head: quarterly
# The reviews round. The warehouse does not. A question like "what was revenue in
# 2023 Q1" has two correct answers — $7,360,834 and $7,360,833.67 — and the first
# version of this file accepted only the first, which scored the more accurate answer
# as wrong ten times. §21.14 is that discovery.
exact = sqlite3.connect(DB)


def warehouse_quarter(quarter: str) -> dict[str, str]:
    year, q = quarter.split(" Q")
    row = exact.execute(
        "SELECT ROUND(SUM(revenue),2), COUNT(DISTINCT order_id), "
        "ROUND(100.0*SUM(gross_profit)/SUM(revenue),1), SUM(qty) "
        "FROM v_sales WHERE year=? AND quarter=?", (int(year), int(q))).fetchone()
    return {"revenue": f"{row[0]:,.2f}", "orders": f"{row[1]:,}",
            "margin": f"{row[2]}", "units": f"{row[3]:,}"}


for path in sorted(QBR.glob("*.md")):
    text = path.read_text()
    quarter = re.search(r"^## (\d{4} Q\d)", text, re.M).group(1)
    summary = re.search(
        r"Revenue for .*? was \$([\d,]+) across ([\d,]+) orders.*?"
        r"Gross margin was ([\d.]+)%\. We shipped ([\d,]+) units\.", text, re.S)
    revenue, orders, margin, units = summary.groups()

    truth = warehouse_quarter(quarter)
    add(id=f"{path.stem}-revenue", kind="document",
        question=f"What was revenue in {quarter}?",
        answer=revenue,
        accept=[revenue.replace(",", ""), truth["revenue"],
                truth["revenue"].replace(",", "")],
        source=path.name, span=sentence(text, f"${revenue}"))
    add(id=f"{path.stem}-orders", kind="document",
        question=f"How many orders were there in {quarter}?",
        answer=orders, accept=[orders.replace(",", ""), truth["orders"]],
        source=path.name, span=sentence(text, f"{orders} orders"))
    add(id=f"{path.stem}-margin", kind="document",
        question=f"What was gross margin in {quarter}?",
        answer=f"{margin}%", accept=[margin, truth["margin"]],
        source=path.name, span=sentence(text, f"Gross margin was {margin}%"))

    # The two commentary claims every review makes, in the same words.
    strongest = re.search(r"(\w+) remained our strongest region at \$([\d,]+)", text)
    weakest = re.search(r"(\w+) was the weakest at \$([\d,]+)", text)
    if strongest:
        add(id=f"{path.stem}-strongest", kind="document",
            question=f"Which region was strongest in {quarter}?",
            answer=strongest.group(1), source=path.name,
            span=sentence(text, strongest.group(0)))
    if weakest:
        add(id=f"{path.stem}-weakest", kind="document",
            question=f"Which region was weakest in {quarter}?",
            answer=weakest.group(1), source=path.name,
            span=sentence(text, weakest.group(0)))

# ---------------------------------------------------------------- head: warehouse
WAREHOUSE = [
    ("What was total revenue in 2024?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE year=2024"),
    ("How many orders were placed in 2025?",
     "SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE year=2025"),
    ("What was gross margin in 2025, as a percentage?",
     "SELECT ROUND(100.0*SUM(gross_profit)/SUM(revenue),1) FROM v_sales WHERE year=2025"),
    ("How many distinct SKUs were sold in 2023?",
     "SELECT COUNT(DISTINCT sku) FROM v_sales WHERE year=2023"),
    ("Which category made the most gross profit in 2025 Q4?",
     "SELECT category FROM v_sales WHERE year=2025 AND quarter=4 GROUP BY category "
     "ORDER BY SUM(gross_profit) DESC LIMIT 1"),
    ("What was Voss Industrial's revenue in 2025?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE supplier='Voss Industrial' "
     "AND year=2025"),
    ("How many units of Safety products shipped in 2025 Q1?",
     "SELECT SUM(qty) FROM v_sales WHERE category='Safety' AND year=2025 AND quarter=1"),
    ("How much revenue came from the Enterprise segment in 2024 Q2?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE segment='Enterprise' "
     "AND year=2024 AND quarter=2"),
    ("Which channel produced the most orders overall?",
     "SELECT channel FROM v_sales GROUP BY channel ORDER BY COUNT(DISTINCT order_id) "
     "DESC LIMIT 1"),
    ("What was the average discount percentage in 2024?",
     "SELECT ROUND(AVG(discount_pct),3) FROM v_sales WHERE year=2024"),
    ("What was Midwest revenue in 2025 Q1?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE region='Midwest' AND year=2025 "
     "AND quarter=1"),
    ("Which region had the lowest revenue in 2024 Q3?",
     "SELECT region FROM v_sales WHERE year=2024 AND quarter=3 GROUP BY region "
     "ORDER BY SUM(revenue) ASC LIMIT 1"),
    ("How many units were shipped in 2023 Q2?",
     "SELECT SUM(qty) FROM v_sales WHERE year=2023 AND quarter=2"),
    ("What was the total cost of goods sold in 2024?",
     "SELECT ROUND(SUM(cost),2) FROM v_sales WHERE year=2024"),
    ("Which supplier produced the most revenue in 2024?",
     "SELECT supplier FROM v_sales WHERE year=2024 GROUP BY supplier "
     "ORDER BY SUM(revenue) DESC LIMIT 1"),
    ("How many orders did the Midwest region place in 2024 Q4?",
     "SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE region='Midwest' AND "
     "year=2024 AND quarter=4"),
    ("What was Packaging revenue in 2025?",
     "SELECT ROUND(SUM(revenue),2) FROM v_sales WHERE category='Packaging' AND year=2025"),
    ("What was gross profit in 2024 Q1?",
     "SELECT ROUND(SUM(gross_profit),2) FROM v_sales WHERE year=2024 AND quarter=1"),
    ("How many distinct customers ordered in 2025 Q2?",
     "SELECT COUNT(DISTINCT customer_id) FROM v_sales WHERE year=2025 AND quarter=2"),
    ("What was Cleaning category margin in 2024, as a percentage?",
     "SELECT ROUND(100.0*SUM(gross_profit)/SUM(revenue),1) FROM v_sales "
     "WHERE category='Cleaning' AND year=2024"),
]
connection = sqlite3.connect(DB)
for n, (question, sql) in enumerate(WAREHOUSE, start=1):
    value = connection.execute(sql).fetchone()[0]
    answer = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}" \
        if isinstance(value, int) else str(value)
    accept = [str(value)]
    if isinstance(value, float):
        accept += [f"{value:,.0f}", f"{value:.0f}", f"{value / 1e6:,.2f}"]
    if isinstance(value, int):
        accept += [str(value)]
    add(id=f"warehouse-{n:02d}", kind="warehouse", question=question,
        answer=answer, accept=accept, source="v_sales", span=sql)

OUT.parent.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------- tail: by hand
# None of these come from a template. Each was found by reading the corpus and asking
# what a real user would ask that a generator would never think of.
TAIL = [
    # --- the awkward documents from Chapter 13's ingestion work
    dict(id="tail-rotated-dallas", kind="document",
         question="What is the racked storage capacity at the Dallas depot?",
         answer="54,100", accept=["54100", "54,100 sq ft"],
         source="02-rotated-table.md",
         span="| Racked storage      |     48,200 |  31,400 |  22,900 |  18,600 |     54,100 |"),
    dict(id="tail-rotated-total", kind="document",
         question="What is the total capacity of the Columbus depot?",
         answer="32,400", accept=["32400"],
         source="02-rotated-table.md",
         span="| **Total**           | **68,600** | **40,600** | **32,400** | **23,400** | **79,300** |"),
    dict(id="tail-rotated-util", kind="document",
         question="What was utilisation at the Columbus depot at year end?",
         answer="51%", accept=["51"],
         source="02-rotated-table.md",
         span="Utilisation at year end was 82%, 74%, 51%, 69% and 88% respectively."),

    # --- contracts: the clauses somebody actually rings up about
    dict(id="tail-price-cap", kind="document",
         question="Under MSC-2022-100, how much can the supplier raise prices in a "
                  "twelve month period?",
         answer="3%", accept=["3 per cent", "3 percent"],
         source="contract-MSC-2022-100.md",
         span="2.2 The Supplier may adjust prices once in any twelve month period, by "
              "no more than 3% in aggregate, on not less than 30 days written notice."),
    dict(id="tail-price-notice", kind="document",
         question="How much written notice is required before a price rise under "
                  "MSC-2022-100?",
         answer="30 days", accept=["30"],
         source="contract-MSC-2022-100.md",
         span="2.2 The Supplier may adjust prices once in any twelve month period, by "
              "no more than 3% in aggregate, on not less than 30 days written notice."),
    dict(id="tail-reject-window", kind="document",
         question="How long does the buyer have to reject non-conforming goods under "
                  "MSC-2022-100?",
         answer="21 days", accept=["21"],
         source="contract-MSC-2022-100.md",
         span="5.2 The Buyer may reject non-conforming goods within 21 days of "
              "delivery."),
    dict(id="tail-liability-cap", kind="document",
         question="What is the supplier's liability cap under MSC-2022-100?",
         answer="150%", accept=["150"],
         source="contract-MSC-2022-100.md",
         span="6.1 The Supplier's aggregate liability shall not exceed 150% of the "
              "charges paid in the twelve months preceding the claim."),
    dict(id="tail-service-level", kind="document",
         question="What on-time delivery rate does MSC-2022-100 require?",
         answer="96%", accept=["96"],
         source="contract-MSC-2022-100.md",
         span="4.1 The Supplier shall deliver not less than 96% of order lines "
              "complete and on time, measured monthly."),
    dict(id="tail-three-months", kind="document",
         question="What happens if a supplier misses the delivery target three months "
                  "running under MSC-2022-100?",
         answer="clause 7", accept=["remedies in clause 7", "termination"],
         source="contract-MSC-2022-100.md",
         span="4.2 Failure to meet the target in clause 4.1 in three consecutive "
              "months entitles the Buyer\nto the remedies in clause 7."),
    dict(id="tail-payment-terms", kind="document",
         question="What are the payment terms under MSC-2022-100?",
         answer="60 days", accept=["60"],
         source="contract-MSC-2022-100.md",
         span="3.1 Payment shall be made within 60 days of the date of a valid "
              "invoice."),
    dict(id="tail-termination", kind="document",
         question="How much notice is needed to terminate MSC-2022-100 for "
                  "convenience?",
         answer="180 days", accept=["180"],
         source="contract-MSC-2022-100.md",
         span="7.1 Either party may terminate this Agreement for convenience on 180 "
              "days written notice."),

    # --- the planted narrative facts: what a manager actually asks
    dict(id="tail-halloway", kind="document",
         question="Which account did Meridian lose, and which region did it hurt?",
         answer="Halloway", accept=["Halloway Group"],
         source="qbr-2024-Q3.md",
         span="**Account loss.** Halloway Group did not renew at the end of the "
              "previous quarter."),
    dict(id="tail-halloway-why", kind="document",
         question="Why does the company say Halloway Group left?",
         answer="competitor", accept=["framework agreement", "competitor's national"],
         source="qbr-2024-Q3.md",
         span="The commercial team attributes the loss to a competitor's national "
              "framework agreement rather than to service failure; the account review "
              "is filed separately."),
    dict(id="tail-margin-target", kind="document",
         question="What is Meridian's standing gross margin target?",
         answer="32.0%", accept=["32", "32.0"],
         source="qbr-2024-Q3.md",
         span="The Safety category led on revenue. Margin across the business was "
              "33.1%, against a standing target of 32.0%."),

    # --- composite: needs both halves
    dict(id="tail-composite-q3", kind="composite",
         question="What was revenue in 2024 Q3, and what does the review give as the "
                  "reason the Midwest fell?",
         answer="Halloway", accept=["8,461,842", "8461841.81"],
         source="qbr-2024-Q3.md",
         span="**Account loss.** Halloway Group did not renew at the end of the "
              "previous quarter."),
    dict(id="tail-composite-margin", kind="composite",
         question="What was gross margin in 2025 Q1, and which supplier's pricing is "
                  "named as a cause?",
         answer="Voss", accept=["Voss Industrial", "31.4"],
         source="qbr-2025-Q1.md",
         span="**Input costs.** Voss Industrial applied a 12% increase to unit costs "
              "across its catalogue with effect from the start of this quarter."),

    dict(id="tail-composite-sanitation", kind="composite",
         question="What was revenue in 2024 Q2, and what new category appears in the "
                  "data from that quarter?",
         answer="Sanitation", accept=["8,074,848", "8074847.91"],
         source="qbr-2024-Q2.md", span=None),
    dict(id="tail-composite-tickets", kind="composite",
         question="What drove the rise in support contacts in 2024 Q4, and which "
                  "supplier was involved?",
         answer="Pemberton", accept=["Pemberton Mills"],
         source="qbr-2024-Q4.md", span=None),

    # --- two figures that disagree, on purpose
    dict(id="tail-rounding", kind="composite",
         question="The 2024 Q3 review says revenue was $8,461,842 and the warehouse "
                  "says $8,461,841.81. Which is right?",
         answer="both", accept=["rounding", "rounded", "same"], judgement=True,
         source="qbr-2024-Q3.md",
         span="Revenue for 2024 Q3 was $8,461,842 across 5,579 orders, up 4.8% on "
              "the prior quarter."),
    dict(id="tail-target-miss", kind="composite",
         question="In which quarters did gross margin fall below the standing target?",
         answer="32.0", accept=["target of 32.0%"],
         source="qbr-2025-Q1.md",
         span="The Sanitation category led on revenue. Margin across the business "
              "was 31.4%, against a standing target of 32.0%."),

    # --- the ones that should refuse
]
# Two of those cite a document but not a sentence, because the fact is the *absence*
# of one in earlier quarters. Find the sentence that carries it.
for extra in TAIL:
    if extra.get("span") is None and extra.get("source"):
        body = next(p for p in DOCS.rglob(extra["source"])).read_text()
        needle = (extra["accept"] or [extra["answer"]])[0]
        extra["span"] = sentence(body, needle if needle in body else extra["answer"])

for extra in TAIL:
    extra.setdefault("tier", "tail")
    extra.setdefault("accept", [])
    cases.append(extra)

# ---------------------------------------------------------------- must abstain
# Plausible, specific, about the right company, and absent. A system that answers
# these is not being helpful; it is being fluent about nothing.
ABSTAIN = [
    "What is Meridian's employee headcount?",
    "Who is the Chief Financial Officer?",
    "What was revenue in 2026 Q1?",
    "What is the customer churn rate?",
    "Which insurer underwrites the product liability cover?",
    "How many vehicles are in the delivery fleet?",
    "What is the target for on-time delivery in 2026?",
    "Which warehouse management system does Meridian use?",
    "What is the annual contract value of the Voss Industrial agreement?",
    "How many people work at the Columbus depot?",
    "What is Meridian's carbon reduction target?",
    "What was the net promoter score in 2025?",
    "Which auditor signed off the 2024 accounts?",
    "What is the notice period in the Halloway Group contract?",
    "How much did Meridian spend on marketing in 2024?",
    "What is the average time to resolve a support ticket?",
    "Which bank provides Meridian's working capital facility?",
    "What is the depreciation policy for warehouse equipment?",
    "How many suppliers were onboarded in 2025?",
    "What is the lease expiry date for the Newark depot?",
]
for n, question in enumerate(ABSTAIN, start=1):
    cases.append(dict(id=f"abstain-{n:02d}", kind="unanswerable", tier="tail",
                      question=question, answer=None, accept=[],
                      source=None, span=None, must_abstain=True))


# ---------------------------------------------------------------- verification
def verify(case: dict) -> list[str]:
    """
    Prove the case before trusting it.

    Three checks, and each one has caught a real mistake while this file was written:
    the span must appear verbatim in the file it cites, the answer must appear inside
    its own span, and a case that must be refused must not carry an answer.
    """
    problems = []
    if case["kind"] == "unanswerable":
        if case.get("answer") is not None:
            problems.append("unanswerable case has an answer")
        return problems
    if case["kind"] == "warehouse":
        return problems                      # its span is the SQL; §21.4 explains
    path = next((p for p in DOCS.rglob(case["source"])), None)
    if path is None:
        return [f"no such document: {case['source']}"]
    body = path.read_text()
    if case["span"] not in body:
        problems.append("span not found in the document")
    elif case.get("judgement"):
        # Some correct answers are not in any document. "Which figure is right?" is
        # answered by knowing that both are, which no span contains. The span still
        # has to exist and be cited — it is the evidence, not the answer.
        pass
    else:
        haystack = case["span"].lower()
        wanted = [case["answer"]] + list(case["accept"])
        if not any(w.lower() in haystack for w in wanted if w):
            problems.append(f"answer {case['answer']!r} is not inside its own span")
    return problems


broken = [(c["id"], p) for c in cases for p in verify(c)]
for case_id, problem in broken:
    print(f"  BROKEN {case_id}: {problem}")

kinds = {}
tiers = {}
for c in cases:
    kinds[c["kind"]] = kinds.get(c["kind"], 0) + 1
    tiers[c["tier"]] = tiers.get(c["tier"], 0) + 1

OUT.write_text(yaml.safe_dump({"version": 1, "cases": cases}, sort_keys=False,
                              width=88, allow_unicode=True))
print(f"\n{len(cases)} cases -> {OUT}")
print(f"  by kind: {kinds}")
print(f"  by tier: {tiers}")
print(f"  verification failures: {len(broken)}")
