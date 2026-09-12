# What the model actually sees. Not letters, not words — tokens.

import tiktoken

from clarity.config import MODEL_FAST

encoder = tiktoken.encoding_for_model(MODEL_FAST)

SAMPLES = [
    "Meridian Supply Co.",
    "unbelievable",
    "MRD-CLE-001",
    "$8,461,842",
    "Halloway",
    "strawberry",
    "  leading spaces matter",
]

for text in SAMPLES:
    ids = encoder.encode(text)
    pieces = [encoder.decode([i]) for i in ids]
    print(f"{text!r}")
    print(f"   {len(ids)} tokens: {pieces}")
    print(f"   ids:        {ids}")
    print()

print("Notice: a token is not a word and not a letter. Common words are one token.")
print("Rare names, product codes and numbers get chopped into pieces.")
