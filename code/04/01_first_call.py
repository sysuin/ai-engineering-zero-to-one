# Your first call to a language model. Six lines of work.

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()          # reads OPENAI_API_KEY from the environment

response = client.chat.completions.create(
    model=MODEL_FAST,
    messages=[{"role": "user", "content": "In one sentence: what is a distributor?"}],
)

print(response.choices[0].message.content)
