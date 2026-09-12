# timeout: 600
# The same request, in two providers' shapes, and what has to be translated.

import json
import sys

sys.path.insert(0, "code")
sys.path.insert(0, "code/27")
from _stub_provider import BlockTransport                      # noqa: E402
from clarity.config import MODEL_FAST                          # noqa: E402
from clarity.platform.gateway import (BlockShapedProvider,     # noqa: E402
                                      Gateway, OpenAIProvider)

MESSAGES = [
    {"role": "system", "content": "You are an analyst for Meridian."},
    {"role": "user", "content": "What was revenue in 2024 Q3?"},
]
TOOLS = [{"type": "function", "function": {
    "name": "query_warehouse",
    "description": "Compute a figure from the sales database.",
    "parameters": {"type": "object",
                   "properties": {"question": {"type": "string"}},
                   "required": ["question"]}}}]

transport = BlockTransport()
providers = {"openai": OpenAIProvider(MODEL_FAST),
             "block-shaped": BlockShapedProvider("stub-1", transport)}

print("One request, two providers, one Reply type.\n")
replies = {}
for name, provider in providers.items():
    reply = Gateway(provider).complete(MESSAGES, tools=TOOLS, max_tokens=200)
    replies[name] = reply
    call = reply.tool_calls[0].name if reply.tool_calls else "-"
    print(f"  {name:<14}stop={reply.stop_reason:<11}tool={call:<17}"
          f"{reply.tokens_in:>4}/{reply.tokens_out:<4} tokens")

print("\nWhat the adapter had to translate to get there:\n")
sent = transport.last_payload
differences = [
    ("the system prompt", "a message with role=system",
     "a top-level argument"),
    ("tool schemas", "nested under \"function\"",
     "flat, with input_schema"),
    ("tool results", "a message with role=tool",
     "a content block on a user turn"),
    ("the reply body", "message.content, a string",
     "a list of typed blocks"),
    ("token usage", "prompt_tokens / completion_tokens",
     "input_tokens / output_tokens"),
    ("the stop reason", "\"stop\" / \"length\" / \"tool_calls\"",
     "\"end_turn\" / \"max_tokens\" / \"tool_use\""),
]
print(f"  {'':<16}{'one shape':<30}{'the other'}")
for what, left, right in differences:
    print(f"  {what:<16}{left[:28]:<30}{right[:26]}")
print(f"\n  (the stand-in received the system prompt as: "
      f"{sent['system'][:24]}…)")

json.dump({"replies": {k: {"stop_reason": v.stop_reason, "tokens_in": v.tokens_in,
                           "tokens_out": v.tokens_out,
                           "tool": v.tool_calls[0].name if v.tool_calls else None}
                       for k, v in replies.items()},
           "differences": differences},
          open("code/27/_gateway.json", "w"), indent=1)

print()
print("Six translations, and not one of them is exotic. That is the argument for the")
print("gateway: none of these differences is hard, and all of them are load-bearing,")
print("so the cost of not having one place to put them is that all six leak into")
print("application code and stay there.")
print()
print("The stop reason is the one worth dwelling on. Both providers can tell you an")
print("answer was cut off, and they use different words for it, and almost every")
print("codebase that has not thought about this ignores the field entirely — which")
print("means a truncated answer is indistinguishable from a finished one all the way")
print("up to the user.")
print()
print("Note what the interface does *not* expose: streaming granularity, provider")
print("error classes, model-specific parameters. Every one of those is a place a")
print("provider leaks through an abstraction that claimed to hide it, and the")
print("discipline is to add them only when a second provider forces you to.")
print()
print("And be honest about what has been tested here. The block-shaped adapter ran")
print("against a stand-in that speaks that wire format, not against a hosted model.")
print("That tests the translation — which is the part that breaks — and it does not")
print("test the model. §27.3 is the part that needs a real account.")
