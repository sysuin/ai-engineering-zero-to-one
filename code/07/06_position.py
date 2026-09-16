# timeout: 900
# Where the question goes. The same instruction, before the document or after it.

import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from _contracts import load, score
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
truth = [c["notice_days"] for c in contracts]
QUESTION = ("How many days written notice must the Supplier give before a price adjustment? "
            "Reply with the number only.")

LAYOUTS = {
    "question, then document":        lambda text: f"{QUESTION}\n\n<contract>\n{text}\n</contract>",
    "document, then question":        lambda text: f"<contract>\n{text}\n</contract>\n\n{QUESTION}",
    "question before AND after":      lambda text: (f"{QUESTION}\n\n<contract>\n{text}\n"
                                                    f"</contract>\n\n{QUESTION}"),
    "document in system, question in user": None,
}


def ask(layout: str, text: str) -> str:
    if LAYOUTS[layout] is None:
        messages = [{"role": "system", "content": f"<contract>\n{text}\n</contract>"},
                    {"role": "user", "content": QUESTION}]
    else:
        messages = [{"role": "user", "content": LAYOUTS[layout](text)}]
    response = client.chat.completions.create(model=MODEL_FAST, temperature=0,
                                              messages=messages, max_completion_tokens=40)
    return (response.choices[0].message.content or "").strip()


print(f"{len(contracts)} contracts, averaging {sum(len(c['text']) for c in contracts) // len(contracts):,} "
      "characters each\n")
print(f"  {'layout':38} {'accuracy':>9} {'number only':>12}")
accuracies = []
for layout in LAYOUTS:
    with ThreadPoolExecutor(max_workers=12) as pool:
        answers = list(pool.map(lambda c: ask(layout, c["text"]), contracts))
    usable = sum(bool(re.fullmatch(r"\d+", a)) for a in answers) / len(answers)
    accuracies.append(score(answers, truth))
    print(f"  {layout:38} {accuracies[-1]:>8.1%} {usable:>11.0%}")

spread = max(accuracies) - min(accuracies)
print(f"\nspread between the best and worst layout: {100 * spread:.1f} points")
if spread < 0.05:
    print("At this length, with a specific question, layout barely matters for accuracy. The")
    print("case for putting the stable document first is then caching (Chapter 5).")
else:
    print("Layout mattered here — measure it on your own documents before choosing.")
