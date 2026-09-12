# timeout: 1200
# When a tool fails, the error message is the next thing the model reads.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
METRICS = ["revenue", "gross_profit", "margin_pct", "orders", "units"]

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
        return (f"Error: unknown metric {metric!r}. Valid metrics are: "
                f"{', '.join(METRICS)}. If you wanted total sales value, use "
                f"'revenue'. Retry with metric='revenue'.")
    if region is not None and region not in REGIONS:
        if style == "terse":
            return "Error: invalid region."
        match = next((r for r in REGIONS if r.lower() == str(region).lower()), None)
        hint = (f" Did you mean {match!r}?" if match
                else f" Valid regions are: {', '.join(REGIONS)}.")
        return f"Error: unknown region {region!r}.{hint} Region names are capitalised."
    return json.dumps({"metric": metric, "region": region, "value": 8461841.81})


# Three questions phrased so the model's first attempt is likely to be wrong.
QUESTIONS = [
    "What were total sales in the midwest in 2024?",
    "Give me the turnover for the south-east region.",
    "What is the profit margin percentage for northeast?",
]


def conversation(question: str, style: str, attempts: int = 4) -> tuple[bool, int]:
    messages = [{"role": "user", "content": question}]
    for attempt in range(1, attempts + 1):
        reply = client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=300,
            messages=messages, tools=TOOL).choices[0].message
        if not reply.tool_calls:
            return False, attempt
        messages.append(reply)
        for call in reply.tool_calls:
            result = run_tool(json.loads(call.function.arguments), style)
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "content": result})
            if not result.startswith("Error"):
                return True, attempt
    return False, attempts


print(f"{'error style':14} {'recovered':>10} {'mean attempts':>15}")
summary = {}
for style in ("terse", "instructive"):
    with ThreadPoolExecutor(max_workers=6) as pool:
        outcomes = list(pool.map(lambda q: conversation(q, style), QUESTIONS))
    ok = sum(o for o, _ in outcomes)
    mean = sum(a for _, a in outcomes) / len(outcomes)
    summary[style] = {"recovered": ok, "n": len(QUESTIONS), "mean_attempts": mean}
    print(f"{style:14} {ok:>7}/{len(QUESTIONS)} {mean:>15.1f}")

print("\nWhat each error says:")
print(f"  terse        {run_tool({'metric': 'sales'}, 'terse')}")
print(f"  instructive  {run_tool({'metric': 'sales'}, 'instructive')[:96]}")
print(f"  terse        {run_tool({'metric': 'revenue', 'region': 'midwest'}, 'terse')}")
print(f"  instructive  "
      f"{run_tool({'metric': 'revenue', 'region': 'midwest'}, 'instructive')[:96]}")

Path("code/16/_errors.json").write_text(json.dumps(summary, indent=2))

print()
print("A tool error is not a log line. It is the next thing the model reads, and it is")
print("the only chance to say what a valid call looks like.")
print()
print("Name the valid values. Suggest the correction. Say what to retry with. It costs")
print("a few tokens once and saves a round trip every time it fires.")
