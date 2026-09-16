# timeout: 1800
# How many tools is too many? Selection accuracy and prompt size as unrelated tools are added,
# and what a handful of overlapping ones does instead.

import statistics
import time
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
REPEATS = 3


def schema(name: str, description: str) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": {"input": {"type": "string"}},
                       "required": ["input"], "additionalProperties": False}}}


REAL = [
    schema("search_documents", "Search Meridian's quarterly reviews, support tickets and internal "
           "notes for a passage. Use for what happened, why, or what someone said. Do NOT use "
           "for numbers that must be exact."),
    schema("query_warehouse", "Compute a metric from the sales database: revenue, margin, orders, "
           "units. Use whenever the answer is a number that must be exactly right."),
    schema("fetch_contract", "Fetch the full text of one supplier agreement by its reference, "
           "e.g. MSC-2022-100."),
    schema("arithmetic", "Evaluate an arithmetic expression over numbers you already have."),
]

# Unrelated tools a large company might plausibly expose: 16 systems, 8 actions each.
SYSTEMS = ["employee records", "payroll runs", "IT assets", "travel bookings", "access badges",
           "training courses", "marketing campaigns", "job candidates", "fleet vehicles",
           "maintenance requests", "meeting-room bookings", "expense claims", "software licences",
           "parking permits", "staff surveys", "visitor logs"]
ACTIONS = [("search_{}", "Search {} by keyword."), ("get_{}", "Fetch one of the {} by id."),
           ("list_recent_{}", "List the most recent {}."), ("count_{}", "Count {} matching a filter."),
           ("export_{}", "Export {} to a spreadsheet."), ("summarise_{}", "Summarise {} over a period."),
           ("annotate_{}", "Add a note to one of the {}."), ("archive_{}", "Archive old {}.")]
UNRELATED = [schema(name.format(system.replace(" ", "_").replace("-", "_")), text.format(system))
             for system in SYSTEMS for name, text in ACTIONS]

# Four tools whose territory genuinely overlaps the real ones.
OVERLAPPING = [
    schema("search_meeting_notes", "Search notes from management meetings, including discussion "
           "of results and performance."),
    schema("query_finance_ledger", "Look up figures in the finance ledger, including revenue and "
           "cost postings."),
    schema("get_supplier_record", "Fetch a supplier's record, including its agreements."),
    schema("calculate", "Work out a numeric answer."),
]

QUESTIONS = [
    ("What was total revenue in 2024 Q3?", "query_warehouse"),
    ("What is our gross margin for 2025?", "query_warehouse"),
    ("How many units of Safety products shipped in 2025 Q1?", "query_warehouse"),
    ("Why did the Midwest region decline?", "search_documents"),
    ("What are customers complaining about?", "search_documents"),
    ("What explanation was given for the margin fall?", "search_documents"),
    ("What is the price cap in MSC-2022-100?", "fetch_contract"),
    ("Show me agreement MSC-2023-109.", "fetch_contract"),
    ("What are the payment terms in MSC-2024-118?", "fetch_contract"),
    ("If revenue is 8461842 and cost is 5661205, what is the margin?", "arithmetic"),
    ("What is 41300 times 3.05?", "arithmetic"),
    ("What is 15% of 8461842?", "arithmetic"),
]

CONDITIONS = {
    "4 real tools": REAL,
    "+ 12 unrelated": REAL + UNRELATED[:12],
    "+ 60 unrelated": REAL + UNRELATED[:60],
    "+ 124 unrelated": REAL + UNRELATED[:124],
    "+ 4 overlapping": REAL + OVERLAPPING,
}


def choose(question: str, tools: list[dict]) -> tuple[str | None, int, float]:
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200, tools=tools,
        tool_choice="required", messages=[{"role": "user", "content": question}])
    calls = response.choices[0].message.tool_calls
    return (calls[0].function.name if calls else None, response.usage.prompt_tokens,
            time.perf_counter() - started)


bare = client.chat.completions.create(
    model=MODEL_FAST, max_completion_tokens=20,
    messages=[{"role": "user", "content": QUESTIONS[0][0]}]).usage.prompt_tokens

print(f"{len(QUESTIONS)} questions x {REPEATS} runs; the same question with no tools is "
      f"{bare} prompt tokens\n")
print(f"  {'condition':17} {'tools':>5} {'correct':>9} {'prompt tokens':>14} {'median s':>9}")
wrong_choices, prompt_tokens = {}, {}
for label, tools in CONDITIONS.items():
    jobs = [q for q in QUESTIONS for _ in range(REPEATS)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda q: (q, choose(q[0], tools)), jobs))
    correct = sum(got == want for (_, want), (got, _, _) in results)
    tokens = statistics.median(t for _, (_, t, _) in results)
    seconds = statistics.median(s for _, (_, _, s) in results)
    print(f"  {label:17} {len(tools):>5} {correct:>4}/{len(jobs):<4} {tokens:>14,.0f} {seconds:>9.2f}")
    wrong_choices[label] = [(q, got) for (q, want), (got, _, _) in results if got != want]
    prompt_tokens[len(tools)] = tokens

per_tool = (prompt_tokens[128] - prompt_tokens[4]) / 124
print(f"\neach unrelated tool added about {per_tool:.0f} prompt tokens to every call")
for label, wrong in wrong_choices.items():
    if wrong:
        print(f"\n{label} — {len(wrong)} wrong choices:")
        for question, got in sorted(set(wrong)):
            print(f"  {question[:58]:60} -> {got}  ({wrong.count((question, got))}x)")
