# timeout: 1200
# A tool description is a prompt. Measured.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

# Four tools with genuinely overlapping territory. The names are deliberately
# uninformative — `search`, `query`, `fetch` and `compute` are all plausible for any of
# these tasks — so that the experiment varies the description and nothing else.
#
# A first version of this listing used names like `query_warehouse`, and both conditions
# scored 100%: the names were doing the work and the descriptions were decoration. That
# is worth knowing on its own. A well-named tool needs less description.
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
    return reply.tool_calls[0].function.name if reply.tool_calls else None


results = {}
for label, descriptions in (("terse", TERSE), ("written properly", WRITTEN)):
    with ThreadPoolExecutor(max_workers=12) as pool:
        picks = list(pool.map(lambda q: chosen(q[0], descriptions), QUESTIONS))
    hits = sum(p == want for p, (_, want) in zip(picks, QUESTIONS))
    results[label] = {"correct": hits, "picks": picks}
    print(f"{label:18} {hits}/{len(QUESTIONS)}  {hits / len(QUESTIONS):.0%}")
    for (question, want), got in zip(QUESTIONS, picks):
        if got != want:
            print(f"     {question[:46]:48} chose {got}, wanted {want}")

Path("code/16/_descriptions.json").write_text(json.dumps(
    {"n": len(QUESTIONS),
     "terse": results["terse"]["correct"],
     "written": results["written properly"]["correct"]}, indent=2))

print()
print("A modest difference, and an honest one: one question in twelve.")
print()
print("The failure is the interesting part. With a terse description, 'What was total")
print("revenue in 2024 Q3?' went to the document search — and that is not unreasonable,")
print("because a quarterly review does discuss revenue. What the written description")
print("adds is the sentence that settles it: 'do NOT use for numbers that must be")
print("exact — those come from query'.")
print()
print("Two things follow. Name tools well: the first version of this listing used names")
print("like query_warehouse, and both conditions scored 100% because the names were")
print("doing the work. And when two tools overlap, write the 'do NOT use for' clause,")
print("because that is the half that disambiguates and the half everyone omits.")
