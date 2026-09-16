# timeout: 900
# Four ways to ask for "just the number", including the prohibition, on forty contracts.

import re
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from _contracts import load, score
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
truth = [c["notice_days"] for c in contracts]
QUESTION = "How many days written notice must the Supplier give before a price adjustment?"

FORMATS = {
    "positive: number only":   "Reply with the number only.",
    "prohibition":             "Do not include any explanation, units or other words.",
    "prohibition, paired":     ("Do not include any explanation. Reply with the number only; "
                                "if the contract states none, reply NONE."),
    "no format instruction":   "",
    "an output template":      "Answer in exactly this form:\nDAYS=<integer>",
}


def ask(fmt: str, text: str) -> str:
    content = f"{QUESTION} {fmt}".strip() + f"\n\n<contract>\n{text}\n</contract>"
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=250,
        messages=[{"role": "user", "content": content}],
    ).choices[0].message.content or "").strip()


print(f"  {'instruction':26} {'accuracy':>9} {'exact form':>11}  a sample answer")
for name, fmt in FORMATS.items():
    with ThreadPoolExecutor(max_workers=12) as pool:
        answers = list(pool.map(lambda c: ask(fmt, c["text"]), contracts))
    pattern = r"DAYS=\d+" if "DAYS=" in fmt else r"\d+"
    exact = sum(bool(re.fullmatch(pattern, a)) for a in answers) / len(answers)
    sample = answers[0].replace("\n", " ")[:28]
    print(f"  {name:26} {score(answers, truth):>8.1%} {exact:>10.0%}  {sample!r}")
