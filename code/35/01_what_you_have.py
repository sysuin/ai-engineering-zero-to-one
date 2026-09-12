# What your account can actually reach, today, from your account — not from this page.
#
# This is the only responsible way for a book to talk about models: print the method,
# not the list. Everything below is read from the API at the moment you run it.

import sys
from datetime import datetime, timezone

sys.path.insert(0, "code")
from openai import OpenAI                                       # noqa: E402

from clarity.config import (CAPABILITIES, MODEL_EMBED,          # noqa: E402
                            MODEL_FAST, MODEL_SMART, rate)

client = OpenAI()
models = sorted(client.models.list(), key=lambda m: m.id)

print(f"{len(models)} models reachable from this account "
      f"on {datetime.now(timezone.utc):%Y-%m-%d}.\n")

FAMILIES = {"chat": ("gpt", "o1", "o3", "o4"), "embeddings": ("text-embedding",),
            "audio": ("whisper", "tts"), "images": ("dall-e", "gpt-image")}


def family(model_id: str) -> str:
    for name, prefixes in FAMILIES.items():
        if any(model_id.startswith(p) for p in prefixes):
            return name
    return "other"


counts = {}
for model in models:
    counts[family(model.id)] = counts.get(family(model.id), 0) + 1
for name, n in sorted(counts.items(), key=lambda kv: -kv[1]):
    print(f"  {name:<14}{n:>4}")

print("\nThe three this book names, and what they resolve to for you:\n")
available = {m.id for m in models}
for label, model in (("MODEL_FAST", MODEL_FAST), ("MODEL_SMART", MODEL_SMART),
                     ("MODEL_EMBED", MODEL_EMBED)):
    here = "reachable" if model in available else "NOT on this account"
    prices = rate(model)
    cost = (f"${prices[0]}/{prices[1]} per Mtok" if prices
            else "no rate set — see below")
    print(f"  {label:<13}{model:<24}{here:<20}{cost}")
    params = sorted(CAPABILITIES.get(model, []))
    print(f"  {'':<13}accepts: {', '.join(params) if params else 'unknown'}")

print("\nRates are not in this book, and that is deliberate. To fill them in:\n")
print("  1. Open your provider's pricing page and read the dollars per million")
print("     tokens for each model above — input and output are different numbers.")
print("  2. Put them in .env as RATE_<MODEL>_INPUT and RATE_<MODEL>_OUTPUT, with")
print("     the model name upper-cased and every - and . turned into _.")
print("  3. Re-run. Every listing in the book that reports tokens will now also")
print("     report dollars, using your rates rather than someone's recollection.")
print()
example = MODEL_FAST.upper().replace("-", "_").replace(".", "_")
print(f"  For {MODEL_FAST}, those two lines are:\n")
print(f"      RATE_{example}_INPUT=0.15")
print(f"      RATE_{example}_OUTPUT=0.60")
print("\n  (0.15 and 0.60 are placeholders standing in for a real pricing page.")
print("   The point is the shape of the line, not the number on it.)")
