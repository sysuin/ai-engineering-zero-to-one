# What a gateway should record: the model requested and the model that answered. An alias is
# resolved by the provider, and the response says to what.

import sys

sys.path.insert(0, "code")
from openai import OpenAI                                          # noqa: E402

from clarity.config import MODEL_FAST, MODEL_SMART                 # noqa: E402

client = OpenAI()
print(f"  {'requested':16}{'served':28}{'fingerprint':>12}")
for model in (MODEL_FAST, MODEL_SMART):
    response = client.chat.completions.create(
        model=model, max_completion_tokens=400,
        messages=[{"role": "user", "content": "Reply: ready."}])
    fingerprint = str(response.system_fingerprint)
    print(f"  {model:16}{response.model:28}{fingerprint:>12}")
print("\nthe served name is what belongs on the span; when it changes and the")
print("requested name did not, the provider moved the alias, and the golden set")
print("is the next thing to run")
