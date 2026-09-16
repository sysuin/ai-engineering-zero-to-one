# timeout: 1800
# A test that needs no expected answer: ask the same question two ways and require the same
# answer. Where the truth is known, it also says which phrasing broke.

import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.v0_6.retrieve import Retriever                     # noqa: E402
from clarity.v0_7.warehouse import Warehouse                    # noqa: E402
from clarity.v0_8.tools import build_tools                      # noqa: E402
from clarity.v0_9.agent import Agent, Budget                    # noqa: E402
from meridian_index import load_index                           # noqa: E402

SYSTEM = ("You are an analyst for Meridian. Use tools for every fact. If the answer is not "
          "in the documents or the warehouse, say so plainly rather than guessing.")
con = sqlite3.connect("file:data/meridian/warehouse/meridian.db?mode=ro", uri=True)


def one(sql: str):
    return con.execute(sql).fetchone()[0]


PAIRS = [
    ("What was revenue in 2024 Q3?",
     "How much did Meridian sell in the third quarter of 2024?",
     one("SELECT SUM(revenue) FROM v_sales WHERE year=2024 AND quarter=3")),
    ("What was gross margin in 2025 Q1?",
     "What share of revenue was gross profit in the first quarter of 2025?",
     one("SELECT 100.0*SUM(gross_profit)/SUM(revenue) FROM v_sales WHERE year=2025 AND quarter=1")),
    ("How many orders were there in 2023 Q2?",
     "What was the number of orders placed in the second quarter of 2023?",
     one("SELECT COUNT(DISTINCT order_id) FROM v_sales WHERE year=2023 AND quarter=2")),
    ("Which region had the highest revenue in 2024?",
     "In 2024, which sales region brought in the most money?",
     one("SELECT region FROM v_sales WHERE year=2024 GROUP BY region ORDER BY SUM(revenue) DESC LIMIT 1")),
    ("What was Voss Industrial's revenue in 2025?",
     "How much revenue came from products supplied by Voss Industrial in 2025?",
     one("SELECT SUM(revenue) FROM v_sales WHERE supplier='Voss Industrial' AND year=2025")),
    ("How many units were sold in 2024 Q4?",
     "What was the unit volume shipped in the last quarter of 2024?",
     one("SELECT SUM(qty) FROM v_sales WHERE year=2024 AND quarter=4")),
    ("What was Midwest revenue in 2024 Q3?",
     "How much did the Midwest region bring in during Q3 2024?",
     one("SELECT SUM(revenue) FROM v_sales WHERE region='Midwest' AND year=2024 AND quarter=3")),
    ("Which category made the most gross profit in 2025?",
     "In 2025, what product category earned the largest gross profit?",
     one("SELECT category FROM v_sales WHERE year=2025 GROUP BY category ORDER BY SUM(gross_profit) DESC LIMIT 1")),
]

chunks, vectors = load_index()
tools = build_tools(Retriever(chunks, vectors), Warehouse())


def states(answer: str, truth) -> bool:
    if isinstance(truth, str):
        return truth.lower() in answer.lower()
    for text, unit in re.findall(r"(\d[\d,]*(?:\.\d+)?)\s*(million|m\b)?",
                                 answer.replace("**", ""), flags=re.I):
        value = float(text.replace(",", "")) * (1e6 if unit else 1)
        if abs(value - truth) <= max(abs(truth) * 0.005, 0.05):
            return True
    return False


def ask(question: str) -> str:
    return Agent(tools, budget=Budget(steps=6)).run(question, system=SYSTEM).answer


questions = [q for a, b, _ in PAIRS for q in (a, b)]
with ThreadPoolExecutor(max_workers=8) as pool:
    answers = dict(zip(questions, pool.map(ask, questions)))

both = neither = split = 0
print(f"{len(PAIRS)} questions, each asked two ways; truth from the warehouse\n")
for plain, paraphrase, truth in PAIRS:
    a, b = states(answers[plain], truth), states(answers[paraphrase], truth)
    both += a and b
    neither += not a and not b
    split += a != b
    mark = "both right" if a and b else "both wrong" if not (a or b) else "DIFFERENT"
    print(f"  {mark:12} {plain}")
    if a != b:
        broken = paraphrase if a else plain
        print(f"               broke when asked: {broken}")
print(f"\nboth right {both}, both wrong {neither}, right one way and wrong the other {split}")
