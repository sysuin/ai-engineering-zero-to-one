# timeout: 300
# A budget is only as good as its count. Counting the text of a prompt locally, and
# comparing with what the provider bills, shows what the local count leaves out.

import json

import tiktoken
from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
encoder = tiktoken.encoding_for_model(MODEL_FAST)

TOOL = {"type": "function", "function": {
    "name": "query_warehouse",
    "description": "Answer a numeric question from Meridian's sales warehouse.",
    "parameters": {"type": "object", "properties": {
        "question": {"type": "string", "description": "The question, in plain words."}},
        "required": ["question"]}}}

turn = [{"role": "user", "content": "What was revenue in 2024 Q3?"},
        {"role": "assistant", "content": "Revenue in 2024 Q3 was $8,461,841.81."}]
CASES = [
    ("one short message", [turn[0]], None),
    ("a system prompt and a question",
     [{"role": "system", "content": "You are an analyst for Meridian."}, turn[0]], None),
    ("ten turns of conversation", turn * 5, None),
    ("one question and one tool", [turn[0]], [TOOL]),
    ("one question and four tools", [turn[0]],
     [{**TOOL, "function": {**TOOL["function"], "name": f"{TOOL['function']['name']}_{i}"}}
      for i in range(4)]),
]


def local(messages, tools) -> int:
    """What most budget code counts: the text of each message, nothing else."""
    return sum(len(encoder.encode(m["content"])) for m in messages)


print(f"  {'prompt':<32} {'text only':>10} {'billed':>7} {'difference':>11}")
for label, messages, tools in CASES:
    extra = {"tools": tools} if tools else {}
    usage = client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=16, messages=messages, **extra).usage
    counted = local(messages, tools)
    print(f"  {label:<32} {counted:>10} {usage.prompt_tokens:>7} "
          f"{usage.prompt_tokens - counted:>+11}")

schema = len(encoder.encode(json.dumps(TOOL)))
print(f"\n  (the tool's JSON schema, as text, is {schema} tokens)")
print("\nEvery message carries a few tokens of framing that its text does not show. Tools")
print("add more than their text: the first brings a fixed preamble with it, and each further")
print("one adds its definition in the provider's own format — on every call, used or not. A")
print("budget that counts only message text is short by both, and the second grows with")
print("each tool. Count from the provider's usage figures once, and budget with the margin.")
