# timeout: 900
# The advice people repeat, tested against a set with known answers.
#
# Run at two difficulties. A well-specified prompt is already at 100%, so nothing
# could show an improvement there; the underspecified one has 87 points of headroom.

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from _contracts import load, score
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
truth = [c["notice_days"] for c in contracts]

TASKS = {
    "well specified":
        "How many days written notice must the Supplier give before a price "
        "adjustment?\n\nReply with the number only, no units, no explanation.\n\n{text}",
    "underspecified":
        "How many days notice?\n\n{text}",
}

VARIANTS = {
    "plain":          "{task}",
    "polite":         "Please, if you would be so kind: {task}\n\nThank you very much.",
    "urgent":         "This is EXTREMELY IMPORTANT and URGENT.\n\n{task}",
    "expert framing": "You are a world-class contract lawyer with 30 years of "
                      "experience.\n\n{task}",
    "offered a tip":  "I will tip you $200 for a correct answer.\n\n{task}",
    "threatened":     "If you get this wrong I will lose my job.\n\n{task}",
    "deep breath":    "Take a deep breath and work through this step by step.\n\n{task}",
}


def ask(prompt: str) -> str:
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0,
        messages=[{"role": "user", "content": prompt}],
        max_completion_tokens=250,
    ).choices[0].message.content or "").strip()


results: dict[str, dict[str, float]] = {}
header = f"{'variant':18}" + "".join(f"{name:>18}" for name in TASKS)
print(header)
print("-" * len(header))

for label, wrapper in VARIANTS.items():
    row = {}
    for task_name, task_template in TASKS.items():
        def build(contract: dict, t=task_template, w=wrapper) -> str:
            return w.format(task=t.format(text=contract["text"]))

        with ThreadPoolExecutor(max_workers=12) as pool:
            answers = list(pool.map(lambda c: ask(build(c)), contracts))
        row[task_name] = score(answers, truth)
    results[label] = row
    print(f"{label:18}" + "".join(f"{v:>17.1%}" for v in row.values()))

Path("code/07/_folklore.json").write_text(json.dumps(results, indent=2))

for task_name in TASKS:
    values = [r[task_name] for r in results.values()]
    print(f"\n{task_name}: spread between best and worst = "
          f"{max(values) - min(values):.1%}")

print()
print(f"{len(contracts)} contracts, both columns. Nothing moves.")
print()
print("The second column matters more than the first. A prompt already at 100% has no")
print("room to improve, so a flat result there proves little. The underspecified prompt")
print("has 87 points of headroom, and politeness, urgency, expertise, tips, threats and")
print("breathing exercises recover none of them.")
print()
print("What does recover them is saying which clause you mean. That is the previous")
print("listing, and it is the whole of prompt engineering in one sentence.")
