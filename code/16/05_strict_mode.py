# timeout: 600
# Strict mode: the arguments are guaranteed to match the schema. What the schema must look
# like for that, and what the guarantee does and does not buy.

import json
from concurrent.futures import ThreadPoolExecutor

from openai import BadRequestError, OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]


def tool(properties: dict, required: list[str], strict: bool, closed: bool = True) -> list[dict]:
    parameters = {"type": "object", "properties": properties, "required": required}
    if closed:
        parameters["additionalProperties"] = False
    return [{"type": "function", "function": {
        "name": "revenue", "strict": strict, "parameters": parameters,
        "description": "Total revenue for one sales region and year."}}]


def ask(question: str, tools: list[dict]) -> str:
    try:
        reply = client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=200, tools=tools,
            tool_choice="required", messages=[{"role": "user", "content": question}])
    except BadRequestError as error:
        message = error.body.get("message", str(error)) if isinstance(error.body, dict) else str(error)
        return "REJECTED: " + message.split("context=(), ")[-1]
    return reply.choices[0].message.tool_calls[0].function.arguments


year = {"type": "integer"}
region = {"type": "string", "enum": REGIONS}
region_or_null = {"type": ["string", "null"], "enum": [*REGIONS, None],
                  "description": "null if the region asked about is not one of these."}

print("What a strict schema must look like:")
for label, schema in (
        ("every field required, closed", tool({"region": region, "year": year}, ["region", "year"], True)),
        ("year left optional", tool({"region": region, "year": year}, ["region"], True)),
        ("additionalProperties omitted",
         tool({"region": region, "year": year}, ["region", "year"], True, closed=False))):
    print(f"  {label}\n     {ask('Midwest revenue in 2024?', schema)}")

CONDITIONS = {
    "not strict, enum":          tool({"region": region, "year": year}, ["region", "year"], False),
    "strict, enum":              tool({"region": region, "year": year}, ["region", "year"], True),
    "strict, enum or null":      tool({"region": region_or_null, "year": year},
                                      ["region", "year"], True),
}
QUESTIONS = {"midwest": "What was midwest revenue in 2024?",
             "Europe": "What was revenue in Europe in 2024?",
             "the Pacific Northwest": "What was revenue in the Pacific Northwest in 2024?"}

print("\nWhat the guarantee buys, for a region that is and two that are not in the list:")
for label, tools in CONDITIONS.items():
    with ThreadPoolExecutor(max_workers=3) as pool:
        replies = list(pool.map(lambda q: ask(q, tools), QUESTIONS.values()))
    print(f"  {label}")
    for asked, arguments in zip(QUESTIONS, replies):
        chosen = json.loads(arguments).get("region") if arguments.startswith("{") else arguments
        verdict = ("in the list" if chosen in REGIONS else "null" if chosen is None
                   else "NOT A VALID VALUE")
        print(f"     asked about {asked!r:24} region={chosen!r:12} {verdict}")
