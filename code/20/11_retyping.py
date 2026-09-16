# timeout: 1200
# One agent's figure, written into a supervisor's answer alongside other figures. How often
# does the number survive the synthesis exactly, and what happens to it when it does not?

import random
import re
import textwrap
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from openai import OpenAI                                          # noqa: E402

from clarity.config import MODEL_FAST                              # noqa: E402

client = OpenAI()
random.seed(20)
FIGURES = 300
SUBJECTS = ["revenue", "freight cost", "the discount total", "order value", "gross profit"]
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]


def make_brief() -> tuple[str, str, float, float]:
    """Four colleagues' notes, eight similar figures, and a question about one of them."""
    notes, seen = [], set()
    while len(notes) < 4:
        subject, region = random.choice(SUBJECTS), random.choice(REGIONS)
        quarter = f"{random.choice([2023, 2024, 2025])} Q{random.randint(1, 4)}"
        if (subject, region, quarter) in seen:          # every question has one right answer
            continue
        seen.add((subject, region, quarter))
        value = round(random.uniform(1_000_000, 9_999_999), 2)
        prior = round(value * random.uniform(0.85, 1.15), 2)
        notes.append((subject, region, quarter, value,
                      f"Analyst: {subject} for the {region} in {quarter} was {value:,.2f}, "
                      f"against {prior:,.2f} the quarter before; gross of discounts."))
    subject, region, quarter, value, line = random.choice(notes)
    prior = float(re.search(r"against ([\d,.]+) the quarter", line).group(1).replace(",", ""))
    brief = "\n".join(n[-1] for n in notes) + (
        "\nResearcher: the documents attribute the movement to an account loss and a "
        "supplier price rise, with freight costs flat.")
    question = f"What was {subject} for the {region} in {quarter}, and why did it move?"
    return brief, question, value, prior


def numbers(text: str) -> list[float]:
    return [float(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\.?\d*", text)
            if n.replace(",", "").replace(".", "").isdigit()]


def synthesise(brief: str, question: str) -> str:
    return client.chat.completions.create(
        model=MODEL_FAST, max_completion_tokens=300,
        messages=[{"role": "system", "content":
                   "You are the supervisor. Your colleagues' notes are below. Write the "
                   "final answer to the user's question in two or three sentences."},
                  {"role": "user", "content": f"{brief}\n\nQuestion: {question}"}]
    ).choices[0].message.content or ""


def verdict(sentence: str, value: float) -> str:
    # The first figure the answer states is the one it is giving as the answer.
    found = numbers(re.sub(r"\b20\d\d\b|\bQ[1-4]\b", "", sentence))[:1]
    if any(abs(n - value) < 0.005 for n in found):
        return "exact"
    if any(abs(n * scale - value) / value < 0.01 for n in found
           for scale in (1, 1_000, 1_000_000)):
        return "rounded"
    if found and "million" not in sentence.lower():
        return "a different number"
    return "rounded" if "million" in sentence.lower() else "missing"


briefs = [make_brief() for _ in range(FIGURES)]
with ThreadPoolExecutor(max_workers=8) as pool:
    sentences = list(pool.map(lambda b: synthesise(b[0], b[1]), briefs))
wrong_quarter = 0

tally: dict[str, int] = {}
examples: dict[str, tuple[float, str]] = {}
for (_, _, value, prior), sentence in zip(briefs, sentences):
    kind = verdict(sentence, value)
    if kind == "a different number" and verdict(sentence, prior) == "exact":
        wrong_quarter += 1
    tally[kind] = tally.get(kind, 0) + 1
    examples.setdefault(kind, (value, sentence))

print(f"{FIGURES} final answers, each written from four colleagues' notes holding")
print("eight similar figures\n")
for kind in ("exact", "rounded", "a different number", "missing"):
    print(f"  {kind:<20} {tally.get(kind, 0):>3}")
    if kind == "a different number" and tally.get(kind):
        print(f"    of which the quarter-before figure from the same note: {wrong_quarter}")
for kind in ("rounded", "a different number"):
    if kind in examples:
        value, sentence = examples[kind]
        print(f"\n  {kind}, for {value:,.2f}:")
        quoted = sentence.strip()[:150]
        for line in textwrap.wrap(quoted, 72):
            print("    " + line)

changed = FIGURES - tally.get("exact", 0)
print(f"\n{changed} of {FIGURES} answers did not state the figure exactly as the warehouse "
      "returned it.")
if changed == 0:
    print(f"None is not zero: with no failures in {FIGURES}, the rate could still be as high as")
    print(f"about 1 in {FIGURES // 3} (the rule of three) — rare enough that a spot check will not")
    print("meet it, common enough that a service answering thousands of questions will.")
else:
    print(f"About 1 in {round(FIGURES / changed)}. Each synthesis is a chance to round, "
          "reformat,")
    print("mistype or pick a neighbour, and a chain of agents takes that chance at every handoff.")
print("A figure that travels as data and is inserted by code arrives exactly by construction,")
print("which no prompt can promise.")
