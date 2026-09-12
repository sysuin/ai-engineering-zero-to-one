# Asking the model how sure it was. There is a number, and you can have it.

import math

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

PROMPTS = [
    ("A question with one clear answer",
     "Meridian's weakest region in 2024 Q3 was the Midwest. "
     "Reply with exactly one word: the region name."),
    ("A question with no single answer",
     "One word: the single most likely reason a supplier misses a delivery date."),
]

for label, prompt in PROMPTS:
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": prompt}],
        temperature=1, logprobs=True, top_logprobs=5, max_completion_tokens=16,
    )
    first = response.choices[0].logprobs.content[0]

    print(f"{label}")
    print(f"  it wrote {response.choices[0].message.content.strip()!r}, "
          f"choosing {first.token!r} first")
    print("  the five candidates it weighed:")
    for alternative in first.top_logprobs:
        probability = math.exp(alternative.logprob)
        bar = "#" * max(1, round(probability * 44))
        print(f"    {alternative.token!r:16} {probability:6.3f}  {bar}")
    print()

print("Every token is a choice among candidates, each with a number attached.")
print("When one candidate is at 1.000 the model is not thinking, it is reciting.")
print("When the top two are close, it was genuinely unsure — and you can measure that")
print("rather than guess at it. Chapter 22 uses this to flag answers for review.")
