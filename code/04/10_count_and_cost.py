# Knowing what a call will cost before you make it.

from pathlib import Path

import tiktoken
from openai import OpenAI

from clarity.config import MODEL_FAST, rate

client = OpenAI()
encoder = tiktoken.encoding_for_model(MODEL_FAST)

review = Path("data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md").read_text()
instruction = "Summarise this quarterly review in three sentences.\n\n"
prompt = instruction + review

# Counting before you send costs nothing and takes no time.
estimated_prompt = len(encoder.encode(prompt))
print(f"Document      {len(review):,} characters")
print(f"Prompt        {estimated_prompt:,} tokens (estimated locally)")
print(f"Ratio         {len(prompt) / estimated_prompt:.1f} characters per token")

response = client.chat.completions.create(
    model=MODEL_FAST,
    messages=[{"role": "user", "content": prompt}],
    max_completion_tokens=200,
)
usage = response.usage

print(f"\nActual, from the response:")
print(f"  prompt_tokens     {usage.prompt_tokens:,}   "
      f"(estimate was off by {usage.prompt_tokens - estimated_prompt:+})")
print(f"  completion_tokens {usage.completion_tokens:,}")

rates = rate(MODEL_FAST)
if rates:
    dollars = (usage.prompt_tokens * rates[0] + usage.completion_tokens * rates[1]) / 1e6
    print(f"\n  cost              ${dollars:.6f}")
    print(f"  all twelve reviews ${dollars * 12:.4f}")
else:
    print("\n  No rates configured, so no dollar figure is shown.")
    print("  cost = (prompt_tokens x input_rate + completion_tokens x output_rate)")
    print("         / 1,000,000")
    print("  Put the two rates from Appendix C into .env to see the number here.")

print(f"\nThe local estimate is a few tokens under because the API adds a small")
print(f"per-message overhead. Close enough to budget with, and free to compute.")
