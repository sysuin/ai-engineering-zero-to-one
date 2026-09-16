# Inside a stream: what the chunks are, where the usage went, and how to get it back.

import time

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
started = time.perf_counter()
chunks, empty, first_at, finish, usage, text = 0, 0, None, None, None, []

stream = client.chat.completions.create(
    model=MODEL_FAST, stream=True, stream_options={"include_usage": True},
    messages=[{"role": "user", "content": "In two sentences: why do suppliers raise prices?"}])

for chunk in stream:
    chunks += 1
    if chunk.usage is not None:                 # the last chunk: usage, and no choices
        usage = chunk.usage
    if not chunk.choices:
        continue
    choice = chunk.choices[0]
    if choice.delta.content:
        if first_at is None:
            first_at = time.perf_counter() - started
        text.append(choice.delta.content)
    else:
        empty += 1
    if choice.finish_reason:
        finish = choice.finish_reason

total = time.perf_counter() - started
print(f"chunks received        {chunks}")
print(f"  carrying text        {len(text)}")
print(f"  carrying no text     {empty}   (the opening role chunk, the finish chunk)")
print(f"  with usage only      {chunks - len(text) - empty}")
print(f"finish_reason          {finish}")
print(f"usage                  prompt {usage.prompt_tokens}, completion {usage.completion_tokens}")
print(f"first text after       {first_at:.2f}s of {total:.2f}s")
print(f"\n{''.join(text)}")
