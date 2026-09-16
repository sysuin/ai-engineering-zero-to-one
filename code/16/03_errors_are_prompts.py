# timeout: 1800
# When a tool fails, the error message is the next thing the model reads. The same
# validation, reported two ways, scored by whether the call that finally succeeded was right.

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
REPEATS, ATTEMPTS = 3, 4

REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
METRICS = ["revenue", "gross_profit", "margin_pct", "orders", "units"]

# No enums, on purpose: this is the tool whose schema did not say, and whose errors must.
TOOL = [{"type": "function", "function": {
    "name": "warehouse",
    "description": "Compute a sales metric, optionally for one region.",
    "parameters": {"type": "object", "properties": {
        "metric": {"type": "string"},
        "region": {"type": "string"},
        "year": {"type": "integer"},
    }, "required": ["metric"], "additionalProperties": False}}}]


def run_tool(args: dict, style: str) -> str:
    """The same validation, reported two ways."""
    metric, region = args.get("metric"), args.get("region")
    if metric not in METRICS:
        if style == "terse":
            return "Error: invalid metric."
        return (f"Error: unknown metric {metric!r}. Valid metrics are: revenue (sales value), "
                "gross_profit, margin_pct (gross margin as a percentage), orders, units "
                "(items sold). Retry with one of these exactly.")
    if region is not None and region not in REGIONS:
        if style == "terse":
            return "Error: invalid region."
        match = next((r for r in REGIONS if r.lower() == str(region).lower()), None)
        hint = (f" Did you mean {match!r}?" if match
                else f" Valid regions are: {', '.join(REGIONS)}.")
        return f"Error: unknown region {region!r}.{hint} Retry with a valid region exactly."
    return json.dumps({"metric": metric, "region": region, "value": 1234567.0})


# Questions phrased the way people ask, so that a first attempt is likely to be invalid.
QUESTIONS = [
    ("What were total sales in the midwest in 2024?", "revenue", "Midwest"),
    ("Give me the turnover for the south-east region.", "revenue", "Southeast"),
    ("What is the profit margin percentage for northeast?", "margin_pct", "Northeast"),
    ("How much gross profit did the west make in 2025?", "gross_profit", "West"),
    ("Number of orders from the south west in 2023?", "orders", "Southwest"),
    ("How many items did we sell in the Mid-West?", "units", "Midwest"),
    ("What were sales volumes in units for the north east?", "units", "Northeast"),
    ("What was the margin % in the SW region?", "margin_pct", "Southwest"),
]


def conversation(item: tuple[str, str, str], style: str) -> tuple[str, int, str]:
    question, want_metric, want_region = item
    messages = [{"role": "user", "content": question}]
    calls = 0
    for _ in range(ATTEMPTS):
        reply = client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=300,
            messages=messages, tools=TOOL).choices[0].message
        if not reply.tool_calls:
            return "gave up", calls, (reply.content or "").strip()
        messages.append(reply)
        for call in reply.tool_calls:
            calls += 1
            args = json.loads(call.function.arguments)
            result = run_tool(args, style)
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if not result.startswith("Error"):
                right = args.get("metric") == want_metric and args.get("region") == want_region
                return ("right" if right else "accepted but wrong"), calls, ""
    return "ran out of attempts", calls, ""


OUTCOMES = ["right", "accepted but wrong", "gave up", "ran out of attempts"]
jobs = [q for q in QUESTIONS for _ in range(REPEATS)]
print(f"{len(QUESTIONS)} questions x {REPEATS} runs, {ATTEMPTS} attempts each\n")
HEADINGS = ["right", "wrong", "gave up", "ran out"]
print(f"  {'error style':12}" + "".join(f"{h:>9}" for h in HEADINGS) + f"{'mean calls':>12}")
summary, gave_up_said = {}, {}
for style in ("terse", "instructive"):
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(lambda q: conversation(q, style), jobs))
    counts = Counter(o for o, _, _ in outcomes)
    mean_calls = sum(c for _, c, _ in outcomes) / len(outcomes)
    summary[style] = {**{o: counts[o] for o in OUTCOMES}, "n": len(jobs), "mean_calls": mean_calls}
    gave_up_said[style] = next((text for o, _, text in outcomes if o == "gave up"), None)
    print(f"  {style:12}" + "".join(f"{counts[o]:>6}/{len(jobs)}" for o in OUTCOMES)
          + f"{mean_calls:>12.1f}")
print("\n  wrong = a valid call with the wrong metric or region, which the tool accepted")

for style, text in gave_up_said.items():
    if text:
        print(f"\nWhen the model gave up after {style} errors, it said, for example:")
        print(f"  \"{' '.join(text.split())[:140]}…\"")

print("\nWhat each error says:")
for args in ({"metric": "sales"}, {"metric": "revenue", "region": "midwest"}):
    for style in ("terse", "instructive"):
        print(f"  {style:12} {run_tool(args, style)}")

Path("code/16/_errors.json").write_text(json.dumps(summary, indent=2))
