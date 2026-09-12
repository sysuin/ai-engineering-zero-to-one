# The second question about the same document is cheaper than the first.

import time
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
review = Path("data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md").read_text()
# Repeat the document so the shared prefix is comfortably past the caching threshold.
document = (review + "\n\n") * 6

QUESTIONS = ["Which region was weakest?",
             "Which category had the lowest margin?",
             "What was gross margin overall?"]

for i, question in enumerate(QUESTIONS, start=1):
    started = time.time()
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            # Identical every time — this is the part that can be reused.
            {"role": "system", "content": "Answer in under ten words, from the "
                                          "document only.\n\n" + document},
            {"role": "user", "content": question},
        ],
        max_completion_tokens=64,
    )
    usage = response.usage
    cached = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
    elapsed = time.time() - started

    print(f"Call {i}: {question}")
    print(f"  prompt tokens {usage.prompt_tokens:,}   "
          f"of which cached {cached:,} ({100 * cached / usage.prompt_tokens:.0f}%)   "
          f"{elapsed:.2f}s")
    print(f"  answer: {(response.choices[0].message.content or '').strip()[:60]}")

print()
print("The document did not change between calls, so the work of reading it did not")
print("have to be repeated. Providers charge less for the reused part, and it returns")
print("faster. The requirement is that the identical text sits at the FRONT of the")
print("prompt — put the question first and you lose the reuse entirely.")
