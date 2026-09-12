# Attention, measured rather than illustrated.
#
# Change one word in a sentence and watch the model's belief about a pronoun flip.
# That is what attention does: it decides which earlier words matter for this one.

import math

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

SENTENCE = ("The pallet would not fit in the van because it was too {}.\n\n"
            "What does 'it' refer to? Reply with exactly one word: pallet or van.")


def first_token_distribution(prompt: str) -> list[tuple[str, float]]:
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": prompt}],
        temperature=0, logprobs=True, top_logprobs=5, max_completion_tokens=16,
    )
    first = response.choices[0].logprobs.content[0]
    return [(t.token, math.exp(t.logprob)) for t in first.top_logprobs]


for word in ("big", "small"):
    print(f'"...because it was too {word}."')
    for token, probability in first_token_distribution(SENTENCE.format(word)):
        bar = "#" * max(1, round(probability * 40))
        print(f"    {token!r:10} {probability:6.3f}  {bar}")
    print()

print("One adjective changed, seven words after the two nouns, and the answer moves.")
print("Nothing about 'pallet' or 'van' changed. What changed is which of them the")
print("model was paying attention to when it read 'it'.")
print()
print("That is the whole idea. Attention is a learned answer to the question:")
print("for this word, which earlier words matter?")
