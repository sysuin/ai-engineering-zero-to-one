# timeout: 1200
# A tool description is a prompt. Measured — and measured against a good name.

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
REPEATS = 3

# Four tools with genuinely overlapping territory. In the first two conditions the names are
# deliberately uninformative — `search`, `query`, `fetch` and `compute` are all plausible for
# any of these tasks — so that only the description varies. The third keeps the terse
# descriptions and gives the tools good names instead.
TERSE = {
    "search": "Search documents.",
    "query": "Query the warehouse.",
    "fetch": "Get a contract.",
    "compute": "Do a calculation.",
}

WRITTEN = {
    "search":
        "Search the text of Meridian's quarterly reviews, support tickets and internal "
        "notes for a passage. Use for questions about what happened, why, or what "
        "someone said. Do NOT use for numbers that must be exact — those come from "
        "`query`.",
    "query":
        "Compute a metric from the sales database: revenue, margin, orders, units. Use "
        "whenever the answer is a number that must be exactly right. Returns the "
        "figure and the definition used. Do NOT use for anything not in the sales "
        "data, such as headcount or contract terms.",
    "fetch":
        "Fetch the full text of one supplier agreement by its reference, e.g. "
        "MSC-2022-100. Use when the question names a specific contract. Do NOT use to "
        "search across contracts — use `search` for that.",
    "compute":
        "Evaluate an arithmetic expression over numbers you already have. Use only "
        "after retrieving the numbers; it cannot look anything up.",
}

NAMED = {"search_documents": TERSE["search"], "query_warehouse": TERSE["query"],
         "fetch_contract": TERSE["fetch"], "arithmetic": TERSE["compute"]}
PLAIN_NAME = dict(zip(NAMED, TERSE))            # search_documents -> search, and so on

QUESTIONS = [
    ("What was total revenue in 2024 Q3?", "query"),
    ("What is our gross margin for 2025?", "query"),
    ("How many units of Safety products shipped in 2025 Q1?", "query"),
    ("Why did the Midwest region decline?", "search"),
    ("What are customers complaining about?", "search"),
    ("What explanation was given for the margin fall?", "search"),
    ("What is the price cap in MSC-2022-100?", "fetch"),
    ("Show me agreement MSC-2023-109.", "fetch"),
    ("What are the payment terms in MSC-2024-118?", "fetch"),
    ("If revenue is 8461842 and cost is 5661205, what is the margin?", "compute"),
    ("What is 41300 times 3.05?", "compute"),
    ("What is 15% of 8461842?", "compute"),
]


def schemas(descriptions: dict[str, str]) -> list[dict]:
    return [{"type": "function", "function": {
        "name": name, "description": text,
        "parameters": {"type": "object",
                       "properties": {"input": {"type": "string"}},
                       "required": ["input"], "additionalProperties": False},
    }} for name, text in descriptions.items()]


def chosen(question: str, descriptions: dict[str, str]) -> str | None:
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200,
        messages=[{"role": "user", "content": question}],
        tools=schemas(descriptions), tool_choice="required",
    ).choices[0].message
    name = reply.tool_calls[0].function.name if reply.tool_calls else None
    return PLAIN_NAME.get(name, name)


CONDITIONS = {"vague names, terse": TERSE, "vague names, written": WRITTEN,
              "good names, terse": NAMED}
jobs = [q for q in QUESTIONS for _ in range(REPEATS)]
results = {}
print(f"{len(QUESTIONS)} questions x {REPEATS} runs, tool choice forced\n")
for label, descriptions in CONDITIONS.items():
    with ThreadPoolExecutor(max_workers=12) as pool:
        picks = list(pool.map(lambda q: chosen(q[0], descriptions), jobs))
    misses = Counter((q, got) for got, (q, want) in zip(picks, jobs) if got != want)
    results[label] = len(jobs) - sum(misses.values())
    print(f"  {label:22} {results[label]:>2}/{len(jobs)}  {results[label] / len(jobs):.0%}")
    for (question, got), count in sorted(misses.items()):
        want = dict(QUESTIONS)[question]
        print(f"       {question[:44]:46} chose {got}, wanted {want} ({count} of {REPEATS})")

Path("code/16/_descriptions.json").write_text(json.dumps(
    {"n": len(jobs), "questions": len(QUESTIONS), "repeats": REPEATS, **results}, indent=2))

terse, written, named = results.values()
print()
if written > terse:
    print(f"Writing the descriptions properly gained {written - terse} of {len(jobs)} choices.")
elif written == terse:
    print("Writing the descriptions properly made no difference to the score.")
else:
    print(f"Writing the descriptions properly lost {terse - written} of {len(jobs)} choices.")
if named >= written:
    print(f"Good names with terse descriptions scored {named}/{len(jobs)}: "
          "the names did at least as much as the descriptions.")
else:
    print(f"Good names with terse descriptions scored {named}/{len(jobs)}, "
          f"below the written descriptions' {written}.")
