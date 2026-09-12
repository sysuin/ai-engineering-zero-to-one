# The model remembers nothing. Every turn resends the entire conversation.

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

messages: list[dict] = [
    {"role": "system", "content": "You are terse. One short sentence per answer."},
]

TURNS = [
    "Meridian's Midwest region lost its largest account in 2024 Q3.",
    "What was the region called again?",
]

for turn, question in enumerate(TURNS, start=1):
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(model=MODEL_FAST, messages=messages)
    answer = response.choices[0].message.content.strip()

    # The reply must go back into the list, or the next turn will not know about it.
    messages.append({"role": "assistant", "content": answer})

    print(f"Turn {turn}")
    print(f"  sent {len(messages) - 1} messages, {response.usage.prompt_tokens} "
          f"prompt tokens")
    print(f"  you:   {question}")
    print(f"  model: {answer}")
    print()

print("Conversation now held in `messages`:")
for message in messages:
    print(f"  {message['role']:9} {message['content'][:62]}")

print("\nThe second turn cost more than the first because it resent the first.")
print("This is why a long conversation gets steadily more expensive.")
