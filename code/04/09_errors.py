# The errors you will meet in week one, triggered on purpose so that you meet them here.

import openai
from openai import OpenAI

from clarity.config import MODEL_FAST, MODEL_SMART

client = OpenAI()
GOOD = [{"role": "user", "content": "Say OK."}]


def attempt(label: str, call) -> None:
    try:
        call()
        print(f"  (no error)  {label}")
    except Exception as error:                              # noqa: BLE001
        name = type(error).__name__
        message = str(error).split("\n")[0][:96]
        print(f"  {name:24} {label}")
        print(f"  {'':24} {message}")


print("1. A key that is not a key")
attempt("authentication", lambda: OpenAI(api_key="sk-not-a-real-key")
        .chat.completions.create(model=MODEL_FAST, messages=GOOD))

print("\n2. A model that does not exist")
attempt("wrong model name", lambda: client.chat.completions.create(
    model="gpt-4o-tubro", messages=GOOD))

print("\n3. A parameter this model does not accept")
attempt("temperature on a reasoning model", lambda: client.chat.completions.create(
    model=MODEL_SMART, messages=GOOD, temperature=0.7))

print("\n4. More text than the model can hold")
attempt("context length", lambda: client.chat.completions.create(
    model=MODEL_FAST,
    messages=[{"role": "user", "content": "supply chain " * 400_000}]))

print("\n5. A message list the API cannot read")
attempt("malformed messages", lambda: client.chat.completions.create(
    model=MODEL_FAST, messages=[{"role": "narrator", "content": "Say OK."}]))

print("\n6. A timeout you set too low")
attempt("timeout", lambda: client.with_options(timeout=0.001)
        .chat.completions.create(model=MODEL_FAST, messages=GOOD))

print("\n7. Too many requests, too fast")
print("  RateLimitError           the one you cannot reliably trigger on purpose")
print("  " + " " * 24 + "and the one you will see most. Chapter 3 is the answer:")
print("  " + " " * 24 + "back off, add jitter, and honour Retry-After.")
