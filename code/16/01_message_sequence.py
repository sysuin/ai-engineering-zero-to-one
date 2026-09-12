# What a tool call actually is: four messages, and no code executed by the model.

import json

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

TOOLS = [{
    "type": "function",
    "function": {
        "name": "gross_margin_pct",
        "description": "Gross margin as a percentage of revenue.",
        "parameters": {
            "type": "object",
            "properties": {
                "revenue": {"type": "number", "description": "Total revenue."},
                "cost": {"type": "number", "description": "Cost of goods sold."},
            },
            "required": ["revenue", "cost"],
            "additionalProperties": False,
        },
    },
}]


def gross_margin_pct(revenue: float, cost: float) -> float:
    """The actual implementation. It runs here, in your process, not anywhere else."""
    return round(100 * (revenue - cost) / revenue, 2)


messages = [{"role": "user", "content":
             "Revenue was 8,461,842 and cost of goods was 5,661,205. "
             "What is the gross margin?"}]

# --- turn one: the model asks for something to be done
first = client.chat.completions.create(
    model=MODEL_FAST, temperature=0, messages=messages, tools=TOOLS)
call = first.choices[0].message.tool_calls[0]

print("1. You send  — the question, plus a description of what is available")
print(f"     {json.dumps(messages[0])[:88]}")
print(f"     tools: [{TOOLS[0]['function']['name']}]\n")

print("2. It replies — not with an answer, with a request")
print(f"     finish_reason  {first.choices[0].finish_reason}")
print(f"     content        {first.choices[0].message.content!r}")
print(f"     tool_calls[0]  id={call.id}")
print(f"                    name={call.function.name}")
print(f"                    arguments={call.function.arguments}\n")

print("3. You run it — the model has executed nothing")
arguments = json.loads(call.function.arguments)
result = gross_margin_pct(**arguments)
print(f"     gross_margin_pct(**{arguments}) -> {result}\n")

# --- turn two: hand the result back, in the message the model is expecting
messages.append(first.choices[0].message)
messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

second = client.chat.completions.create(
    model=MODEL_FAST, temperature=0, messages=messages, tools=TOOLS)

print("4. You send it back — as a 'tool' message, keyed to the call id")
print(f"     {{'role': 'tool', 'tool_call_id': {call.id!r}, 'content': {str(result)!r}}}\n")
print("5. It answers")
print(f"     {second.choices[0].message.content}\n")

print(f"Two round trips, {len(messages) + 1} messages, and the only thing that ran was")
print("your own function. 'Function calling' is a poor name: the model never calls")
print("anything. It emits a structured request and waits to be told what happened.")
