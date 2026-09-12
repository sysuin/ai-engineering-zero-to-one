# What actually came back. Most people read one field of this and ignore the rest.

import json

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

response = client.chat.completions.create(
    model=MODEL_FAST,
    messages=[{"role": "user", "content": "Name three categories a janitorial "
                                          "supplies distributor might sell."}],
)

print("id             ", response.id)
print("model          ", response.model)          # note: more specific than we asked for
print("created        ", response.created)
print("choices        ", len(response.choices))
print()

choice = response.choices[0]
print("finish_reason  ", choice.finish_reason)    # why it stopped
print("role           ", choice.message.role)
print("content        ", repr(choice.message.content[:70]) + " ...")
print()

usage = response.usage
print("prompt_tokens    ", usage.prompt_tokens)      # what you sent, and pay for
print("completion_tokens", usage.completion_tokens)  # what it wrote, and pay more for
print("total_tokens     ", usage.total_tokens)
print()

print("The whole thing, as the dictionary it really is:")
print(json.dumps(response.model_dump(), indent=2, default=str)[:520] + "\n  ...")
