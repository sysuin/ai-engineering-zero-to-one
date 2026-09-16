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


tops = {}
for word in ("big", "small"):
    print(f'"...because it was too {word}."')
    distribution = first_token_distribution(SENTENCE.format(word))
    tops[word] = distribution[0][0]
    for token, probability in distribution:
        bar = "#" * max(1, round(probability * 40))
        print(f"    {token!r:10} {probability:6.3f}  {bar}")
    print()

between = SENTENCE.split("van")[1].split("{}")[0].split()
print(f"One adjective changed, {len(between)} words after the last noun. The model's first "
      f"choice {'moved' if tops['big'] != tops['small'] else 'did not move'}: "
      f"{tops['big']!r} then {tops['small']!r}.")
print("Nothing about 'pallet' or 'van' changed. What changed is which of them the")
print("model was paying attention to when it read 'it'.")
