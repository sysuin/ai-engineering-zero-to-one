# timeout: 900
# Without strict mode, how often does a model send a value outside a tool's enum? Forty ways of
# naming a region — Meridian's five, spelled the ways people spell them, and places that are not
# Meridian regions at all — sent twice each to the same tool, strict mode off.

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
TOOL = [{"type": "function", "function": {
    "name": "revenue", "strict": False,
    "description": "Total revenue for one sales region and year.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["region", "year"],
                   "properties": {"region": {"type": "string", "enum": REGIONS},
                                  "year": {"type": "integer"}}}}}]
SPELLINGS = ["northeast", "North-East", "NE", "the north east", "Mid-West", "midwest", "the Midwest region",
             "MW", "south east", "S.E.", "Southeast region", "the south-east", "west", "the West Coast",
             "western region", "W", "south west", "SW", "Southwest region", "the south-west"]
ELSEWHERE = ["Europe", "Texas", "Canada", "APAC", "the Pacific Northwest", "New England", "Central",
             "North", "South", "Mexico", "the Great Lakes", "EMEA", "Florida", "the Rockies", "LATAM",
             "Northern California", "the UK", "Mid-Atlantic", "Alaska", "online"]


def region_sent(place: str) -> str:
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200, tools=TOOL,
        tool_choice="required",
        messages=[{"role": "user", "content": f"What was revenue in {place} in 2024?"}])
    return json.loads(reply.choices[0].message.tool_calls[0].function.arguments).get("region")


print(f"{len(SPELLINGS)} spellings of Meridian's regions and {len(ELSEWHERE)} places that are not "
      "regions,\neach asked twice, strict mode off\n")
print(f"  {'asked about':<34}{'in the enum':>12}{'outside it':>12}")
outside_values = Counter()
for label, places in (("a Meridian region, spelled loosely", SPELLINGS),
                      ("somewhere that is not a region", ELSEWHERE)):
    with ThreadPoolExecutor(max_workers=10) as pool:
        sent = list(pool.map(region_sent, [p for p in places for _ in range(2)]))
    outside = [s for s in sent if s not in REGIONS]
    outside_values.update(outside)
    print(f"  {label:<34}{len(sent) - len(outside):>8}/{len(sent)}{len(outside):>8}/{len(sent)}")
if outside_values:
    print("\n  values sent that the enum does not allow: "
          + ", ".join(f"{v!r} x{n}" for v, n in outside_values.most_common()))
