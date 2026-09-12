# Temperature: the same prompt, three settings, three samples each.

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

PROMPT = "Give a three-word name for a warehouse restocking alert. Name only."


def sample(temperature: float, n: int = 3) -> list[str]:
    out = []
    for _ in range(n):
        response = client.chat.completions.create(
            model=MODEL_FAST,
            messages=[{"role": "user", "content": PROMPT}],
            temperature=temperature,
        )
        out.append(response.choices[0].message.content.strip())
    return out


for temperature in (0.0, 1.0, 1.6):
    answers = sample(temperature)
    print(f"temperature = {temperature}")
    for answer in answers:
        print(f"    {answer}")
    print(f"    distinct: {len(set(answers))} of {len(answers)}\n")

print("Low temperature narrows the model to its most likely next word.")
print("High temperature lets it wander. Neither is 'more correct'.")
