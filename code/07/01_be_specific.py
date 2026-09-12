# timeout: 600
# The same question, asked four ways. Only the wording changes.

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from _contracts import load, score
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
truth = [c["notice_days"] for c in contracts]

# Every contract contains several "N days" figures: payment terms, notice of price
# change, notice of termination, and a window for rejecting goods. Only one of them
# answers the question, and a vague prompt has no way to know which.
PROMPTS = {
    "vague":
        "How many days notice?\n\n{text}",

    "specific":
        "How many days written notice must the Supplier give before a price "
        "adjustment?\n\n{text}",

    "specific + format":
        "How many days written notice must the Supplier give before a price "
        "adjustment?\n\nReply with the number only, no units, no explanation.\n\n{text}",

    "specific + format + where":
        "Clause 2 of this agreement governs price adjustment. How many days written "
        "notice must the Supplier give before a price adjustment?\n\nReply with the "
        "number only, no units, no explanation.\n\n{text}",
}


def ask(prompt: str) -> tuple[str, int]:
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=250,
    )
    return ((response.choices[0].message.content or "").strip(),
            response.usage.completion_tokens)


results = {}
print(f"{'':32} {'accuracy':>9} {'usable':>8} {'out tokens':>11}")
for label, template in PROMPTS.items():
    with ThreadPoolExecutor(max_workers=12) as pool:
        pairs = list(pool.map(lambda c: ask(template.format(text=c["text"])), contracts))
    answers = [a for a, _ in pairs]
    tokens = sum(t for _, t in pairs)

    accuracy = score(answers, truth)
    # "Usable" means a program could take the answer as-is: it is just a number.
    usable = sum(bool(re.fullmatch(r"\d+", a)) for a in answers) / len(answers)

    print(f"{label:32} {accuracy:>8.1%} {usable:>8.0%} {tokens / len(answers):>11.1f}")
    results[label] = {"accuracy": accuracy, "usable": usable,
                      "tokens": tokens / len(answers), "example": answers[0][:70]}

print()
for label, r in results.items():
    print(f"{label:32} {r['example']!r}")

Path("code/07/_specificity.json").write_text(json.dumps(results, indent=2))

print()
print(f"{len(contracts)} contracts. Same task, same model, same documents — only the")
print("words differ. The vague prompt answers a different question: it finds the")
print("termination notice, which is also 'days notice', and is not what was asked.")
