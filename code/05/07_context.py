# The context window: the model's entire world, and what happens at the edge.

import tiktoken
from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
encoder = tiktoken.encoding_for_model(MODEL_FAST)

# Grow the prompt until the API refuses, so the limit is discovered rather than assumed.
CHUNK = "Meridian Supply Co. shipped pallets of cleaning cloths to five regions. "
sizes = [1_000, 50_000, 400_000]

for words in sizes:
    text = CHUNK * (words // len(CHUNK.split()))
    tokens = len(encoder.encode(text))
    try:
        response = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "user", "content": text + "\n\nReply with one word: OK."}],
            max_completion_tokens=16,
        )
        print(f"  {tokens:>8,} tokens  ->  accepted "
              f"({response.usage.prompt_tokens:,} counted by the API)")
    except Exception as error:                                   # noqa: BLE001
        message = str(error).split("'message': ")[-1][:96]
        print(f"  {tokens:>8,} tokens  ->  refused: {message}")

print()
print("The window holds everything: your system message, the whole conversation so")
print("far, every document you retrieved, and the answer being written. It is a")
print("budget you spend, not a limit you occasionally bump into — and Chapter 10 is")
print("about spending it well.")
