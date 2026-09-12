# Running out of room mid-sentence, and the field that tells you it happened.

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

PROMPT = "Explain what a supply chain is, in about 150 words."

for limit in (40, 400):
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": PROMPT}],
        max_completion_tokens=limit,
    )
    choice = response.choices[0]
    text = (choice.message.content or "").strip()

    print(f"max_completion_tokens = {limit}")
    print(f"  finish_reason     {choice.finish_reason}")
    print(f"  completion_tokens {response.usage.completion_tokens}")
    print(f"  ends with         ...{text[-52:]!r}")
    print()

print("finish_reason == 'length' means you were cut off, not that the model finished.")
print("Check it. A truncated JSON object will not parse, and Chapter 8 depends on it.")
