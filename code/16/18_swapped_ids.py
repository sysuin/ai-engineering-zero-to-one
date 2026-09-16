# timeout: 900
# What does a model do when tool results come back under each other's ids? Ten questions that need
# two regions' revenue, answered with the results swapped: once as bare numbers, and once as results
# that say which region they describe.

import json
import random
from concurrent.futures import ThreadPoolExecutor

import sys
from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                 # noqa: E402

client = OpenAI()
RUNS = 2
TOOLS = [{"type": "function", "function": {
    "name": "revenue", "description": "Total revenue for one sales region and year.",
    "parameters": {"type": "object", "properties": {"region": {"type": "string"},
                                                    "year": {"type": "integer"}},
                   "required": ["region", "year"], "additionalProperties": False}}}]
PAIRS = [("Midwest", "West"), ("Northeast", "Southeast"), ("Southwest", "Midwest"),
         ("West", "Northeast"), ("Southeast", "Southwest")]
rng = random.Random(16)
FIGURES = {r: rng.randint(3_000_000, 9_000_000) for r in ("Midwest", "West", "Northeast",
                                                             "Southeast", "Southwest")}


def ask(a: str, b: str, labelled: bool) -> str:
    question = f"Which had more revenue in 2024, {a} or {b}, and by how much?"
    messages = [{"role": "user", "content": question}]
    reply = client.chat.completions.create(model=MODEL_FAST, messages=messages, tools=TOOLS,
                                           tool_choice="required", parallel_tool_calls=True)
    calls = reply.choices[0].message.tool_calls
    regions = [json.loads(c.function.arguments)["region"] for c in calls]
    if len(calls) != 2 or set(regions) != {a, b}:
        return "did not call once per region"
    results = [{"region": r, "year": 2024, "revenue": FIGURES[r]} if labelled
               else {"revenue": FIGURES[r]} for r in regions]
    messages.append(reply.choices[0].message)
    for call, result in zip(calls, reversed(results)):           # each id gets the other's result
        messages.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result)})
    final = client.chat.completions.create(model=MODEL_FAST, messages=messages).choices[0].message
    text = (final.content or "").lower()
    winner_truth = a if FIGURES[a] > FIGURES[b] else b
    loser_truth = b if winner_truth == a else a
    if any(w in text for w in ("mismatch", "swapped", "inconsistent", "does not match", "doesn't match",
                               "labelled", "labeled", "appears to be for")):
        return "noticed the mismatch"
    said_winner = text.find(winner_truth.lower()), text.find(loser_truth.lower())
    first = winner_truth if 0 <= said_winner[0] < (said_winner[1] if said_winner[1] >= 0 else 10**9) \
        else loser_truth
    return "named the right region" if first == winner_truth else "named the wrong region"


OUTCOMES = ("named the right region", "named the wrong region", "noticed the mismatch",
            "did not call once per region")
print(f"{len(PAIRS)} questions x {RUNS} runs, each answered with the two results swapped\n")
print(f"  {'results':<26}{'right region':>15}{'wrong region':>15}{'noticed':>10}")
for labelled in (False, True):
    jobs = [(a, b) for a, b in PAIRS for _ in range(RUNS)]
    with ThreadPoolExecutor(max_workers=10) as pool:
        outcomes = list(pool.map(lambda p: ask(p[0], p[1], labelled), jobs))
    name = "say which region they are" if labelled else "bare numbers"
    print(f"  {name:<26}" + "".join(f"{outcomes.count(o):>{w}}/{len(jobs)}"
                                    for o, w in zip(OUTCOMES[:3], (12, 12, 7)))
          + (f"   ({outcomes.count(OUTCOMES[3])} runs did not call per region)"
             if outcomes.count(OUTCOMES[3]) else ""))
print("\n  'named' = the first region the answer names as having more revenue; the true figures")
print("  decide which is right")
