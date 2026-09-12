# Three roles. The model treats them differently, and the difference is the whole
# reason you can build a product on top of a chat interface.

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

QUESTION = "Revenue fell 53% in one quarter. What should I look at first?"


def ask(messages: list[dict]) -> str:
    response = client.chat.completions.create(model=MODEL_FAST, messages=messages)
    return response.choices[0].message.content.strip()


print("No system message")
print(" ", ask([{"role": "user", "content": QUESTION}])[:200], "...")

print("\nWith a system message")
print(" ", ask([
    {"role": "system", "content":
        "You are a blunt financial analyst. Answer in at most two sentences. "
        "Never speculate about causes you have no evidence for."},
    {"role": "user", "content": QUESTION},
])[:200], "...")

print("\nWith an assistant turn in the history — the model continues the conversation")
print(" ", ask([
    {"role": "system", "content": "You are a blunt financial analyst."},
    {"role": "user", "content": QUESTION},
    {"role": "assistant", "content": "Check whether a single large account churned."},
    {"role": "user", "content": "It did. What next?"},
])[:200], "...")
