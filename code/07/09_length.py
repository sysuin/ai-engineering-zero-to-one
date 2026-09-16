# timeout: 600
# Asking for a length. How closely does "in N words" get you N words?

import re
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
review = Path("data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md").read_text()
RUNS = 6

PHRASINGS = {
    "in about {n} words":       "Summarise this quarterly review in about {n} words.",
    "in no more than {n} words": "Summarise this quarterly review in no more than {n} words.",
    "in {s} sentences":         "Summarise this quarterly review in {s} sentences.",
}


def count_words(text: str) -> int:
    return len(re.findall(r"\b[\w'’$%.,-]+\b", text))


def run(prompt: str) -> int:
    text = client.chat.completions.create(
        model=MODEL_FAST, temperature=1,
        messages=[{"role": "user", "content": f"{prompt}\n\n{review}"}],
    ).choices[0].message.content or ""
    return count_words(text)


print(f"{RUNS} runs each at temperature 1\n")
print(f"  {'asked for':32} {'target':>6} {'mean':>6} {'min':>5} {'max':>5} {'mean/target':>12}")
for label, template in PHRASINGS.items():
    for n in (30, 120):
        s = max(1, round(n / 20))
        prompt = template.format(n=n, s=s)
        with ThreadPoolExecutor(max_workers=RUNS) as ex:
            counts = list(ex.map(run, [prompt] * RUNS))
        shown = label.format(n=n, s=s)
        print(f"  {shown:32} {n:>6} {statistics.mean(counts):>6.0f} {min(counts):>5} "
              f"{max(counts):>5} {statistics.mean(counts) / n:>11.2f}x")
