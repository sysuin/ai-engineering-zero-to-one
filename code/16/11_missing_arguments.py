# timeout: 900
# A required argument the question does not supply. Does the model ask, send null, or make
# one up? Three schemas for the same tool, on questions with details missing.

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
RUNS = 3

# (question, the region it states, the year it states); None = not stated, or not determinable
QUESTIONS = [
    ("What was Southwest revenue in 2024?", "Southwest", 2024),
    ("Midwest revenue in 2023, please.", "Midwest", 2023),
    ("What was Midwest revenue?", "Midwest", None),
    ("Revenue for the Northeast, please.", "Northeast", None),
    ("What was revenue in 2024?", None, 2024),
    ("Give me revenue.", None, None),
    ("How did the West do last year?", "West", None),
    ("How much did we sell in the south in 2025?", None, 2025),
    ("What was revenue in Europe in 2024?", None, 2024),
    ("What was Midwest revenue in the year we launched Sanitation?", "Midwest", None),
]


def tool(nullable: bool) -> list[dict]:
    region = {"type": ["string", "null"], "enum": REGIONS + [None],
              "description": "The region the question names. null if it names none, or one not in "
                             "the list, or could mean more than one. Never guess."} if nullable else \
             {"type": "string", "enum": REGIONS}
    year = {"type": ["integer", "null"],
            "description": "The year the question states. null if it does not state one. Never guess."} \
        if nullable else {"type": "integer"}
    return [{"type": "function", "function": {
        "name": "revenue", "strict": True,
        "description": "Total revenue for one sales region and year, from the sales database.",
        "parameters": {"type": "object", "additionalProperties": False, "required": ["region", "year"],
                       "properties": {"region": region, "year": year}}}}]


ASK = ("If the question does not say which region and which year it means, ask the user for "
       "what is missing instead of calling a tool.")
CONDITIONS = {"required, no escape": (tool(False), None),
              "null allowed": (tool(True), None),
              "null allowed, told to ask": (tool(True), ASK)}


def outcome(question, region, year, tools, system) -> tuple[str, str]:
    messages = ([{"role": "system", "content": system}] if system else []) + \
               [{"role": "user", "content": question}]
    reply = client.chat.completions.create(model=MODEL_FAST, temperature=0, max_completion_tokens=300,
                                           tools=tools, messages=messages).choices[0].message
    if not reply.tool_calls:
        return "no call", ""
    args = json.loads(reply.tool_calls[0].function.arguments)
    invented = [f"{k}={args[k]}" for k, stated in (("region", region), ("year", year))
                if stated is None and args.get(k) is not None]
    wrong = [f"{k}={args[k]}" for k, stated in (("region", region), ("year", year))
             if stated is not None and args.get(k) != stated]
    if invented or wrong:
        return "invented" if invented else "wrong", ", ".join(invented + wrong)
    return ("right" if region is not None and year is not None else "null"), ""


jobs = [(name, q) for name in CONDITIONS for q in QUESTIONS for _ in range(RUNS)]
with ThreadPoolExecutor(max_workers=12) as pool:
    results = list(pool.map(lambda job: (job, outcome(*job[1], *CONDITIONS[job[0]])), jobs))

print(f"{len(QUESTIONS)} questions x {RUNS} runs; 2 state both details, 8 leave at least one out\n")
print(f"  {'':28}{'complete: right':>16}   {'incomplete:':<12}{'invented':>9}{'null':>6}{'no call':>9}")
examples = {}
for name in CONDITIONS:
    mine = [(q, o) for (n, q), o in results if n == name]
    complete = Counter(o[0] for q, o in mine if q[1] is not None and q[2] is not None)
    incomplete = Counter(o[0] for q, o in mine if q[1] is None or q[2] is None)
    n_c, n_i = 2 * RUNS, 8 * RUNS
    print(f"  {name:28}{complete['right']:>11}/{n_c:<4}   {'':12}{incomplete['invented']:>6}/{n_i}"
          f"{incomplete['null']:>6}{incomplete['no call']:>9}")
    examples[name] = sorted({f"{q[0]} -> {o[1]}" for q, o in mine if o[0] == "invented"})

for name, items in examples.items():
    if items:
        print(f"\ninvented by '{name}':")
        for item in items[:6]:
            print(f"  {item}")
