# timeout: 900
# How many examples is enough? Measured, not guessed.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from openai import OpenAI

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()

# Meridian writes payment terms in an internal shorthand: NET45-INV means "45 days from
# the invoice date", NET30-RCP means "30 days from receipt". The model has no way to
# know this convention. It has to be shown, which is what few-shot prompting is for.
DEMOS, TEST = contracts[:8], contracts[8:]

INSTRUCTION = ("Read the supply agreement and output its payment term in Meridian's "
               "internal shorthand. Reply with the code only.")


def build_messages(contract: dict, n_examples: int) -> list[dict]:
    messages = [{"role": "system", "content": INSTRUCTION}]
    for demo in DEMOS[:n_examples]:
        messages.append({"role": "user", "content": demo["text"]})
        messages.append({"role": "assistant", "content": demo["payment_code"]})
    messages.append({"role": "user", "content": contract["text"]})
    return messages


def ask(contract: dict, n_examples: int) -> str:
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0,
        messages=build_messages(contract, n_examples),
        max_completion_tokens=32,
    ).choices[0].message.content or "").strip()


results = {}
print(f"{'examples':>9} {'accuracy':>9} {'prompt tokens':>14}  first answer")
for n in (0, 1, 2, 4, 8):
    with ThreadPoolExecutor(max_workers=12) as pool:
        answers = list(pool.map(lambda c: ask(c, n), TEST))
    correct = sum(a == c["payment_code"] for a, c in zip(answers, TEST))
    accuracy = correct / len(TEST)

    # What the examples cost you, on every call, forever. Counted locally, for free,
    # exactly as Chapter 4 did.
    encoder = tiktoken.encoding_for_model(MODEL_FAST)
    prompt_tokens = sum(len(encoder.encode(m["content"]))
                        for m in build_messages(TEST[0], n))

    print(f"{n:>9} {accuracy:>8.1%} {prompt_tokens:>14,}  {answers[0]!r}")
    results[n] = {"accuracy": accuracy, "prompt_tokens": prompt_tokens,
                  "example": answers[0]}

Path("code/07/_fewshot.json").write_text(json.dumps(results, indent=2))

print(f"\n{len(TEST)} contracts held out, {len(DEMOS)} available as examples.")
print("The examples, in the order they are added:")
for i, demo in enumerate(DEMOS, start=1):
    print(f"  {i}. {demo['payment_code']}")

print()
print("Read the curve against that list.")
print()
print("With none, the model cannot know the convention exists, so it invents something")
print("reasonable that is not yours.")
print()
print("With one and two, it has the structure but has only ever seen an invoice-based")
print("term, so for a receipt-based contract it has to guess the abbreviation: "
      f"{results[1]['example']!r},")
print(f"then {results[2]['example']!r}.")
print()
print("The jump at four is not about quantity. Example three is the first receipt-based")
print("one. What changed is coverage of the answer space, not the count.")
print()
gap = round((results[4]["accuracy"] - results[8]["accuracy"]) * len(TEST))
ratio = results[8]["prompt_tokens"] / results[4]["prompt_tokens"]
print(f"Eight scores {gap} test item(s) worse than four, out of {len(TEST)}. That is")
print(f"noise, not a finding. The honest reading is that it stopped helping while the")
print(f"prompt got {ratio:.1f}x more expensive — paid on every call, for as long as the")
print("system runs. Chapter 22 is about telling noise from a finding properly.")
