# Two tiers of model on a problem that needs several steps.

import time

from openai import OpenAI

from clarity.config import MODEL_FAST, MODEL_SMART

client = OpenAI()

PROBLEM = (
    "Meridian sells cloths at $12.50 each, costing $8.20. In 2024 Q3 the Midwest "
    "shipped 41,300 cloths at a 10% discount, and the Southwest shipped 28,900 at "
    "full price. Which region made more gross profit, and by how much? "
    "Give the two profit figures and the difference."
)

for model in (MODEL_FAST, MODEL_SMART):
    started = time.time()
    response = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": PROBLEM}])
    elapsed = time.time() - started
    usage = response.usage
    reasoning = getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0

    print(f"{model}")
    print(f"  {elapsed:5.1f}s   {usage.completion_tokens:>5,} completion tokens, "
          f"of which {reasoning:,} were thinking the reader never sees")
    print(f"  {(response.choices[0].message.content or '').strip()[:200]}")
    print()

# The answer, computed the way Clarity will always compute it: in code.
midwest = 41_300 * (12.50 * 0.9 - 8.20)
southwest = 28_900 * (12.50 - 8.20)
print(f"Computed in Python:  Midwest ${midwest:,.0f}   "
      f"Southwest ${southwest:,.0f}   difference ${abs(midwest - southwest):,.0f}")
print()
print("A reasoning model spends tokens thinking before it answers. It is slower and")
print("dearer, and on arithmetic it is still a language model. The right answer to")
print("this question is not a better model — it is to compute the number in code and")
print("let the model describe it. That rule shapes the whole of Clarity.")
