# timeout: 600
# Who decides whether a tool is called: tool_choice, and parallel_tool_calls.

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
TOOLS = [{"type": "function", "function": {
    "name": "revenue", "strict": True,
    "description": "Total revenue for one sales region and year, from the sales database.",
    "parameters": {"type": "object", "additionalProperties": False, "required": ["region", "year"],
                   "properties": {"region": {"type": "string", "enum": REGIONS},
                                  "year": {"type": "integer"}}}}}]


def first_reply(question: str, **options):
    return client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300, tools=TOOLS,
        messages=[{"role": "user", "content": question}], **options).choices[0].message


def describe(message) -> str:
    if message.tool_calls:
        return "; ".join(f"{c.function.name}({c.function.arguments})" for c in message.tool_calls)
    return f"text: {(message.content or '').strip()[:60]!r}"


CHOICES = {"auto": "auto", "none": "none", "required": "required",
           "one named function": {"type": "function", "function": {"name": "revenue"}}}

for question in ("What is the capital of France?", "What was Midwest revenue in 2024?"):
    print(question)
    for label, choice in CHOICES.items():
        print(f"  tool_choice={label:20} {describe(first_reply(question, tool_choice=choice))}")
    print()

both = "What was revenue in 2024 for the Midwest and for the West?"
print(both)
for parallel in (True, False):
    message = first_reply(both, parallel_tool_calls=parallel)
    print(f"  parallel_tool_calls={parallel!s:5}  {len(message.tool_calls or [])} call(s) "
          f"in the first reply")
