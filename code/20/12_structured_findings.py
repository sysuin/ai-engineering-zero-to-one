# timeout: 1200
# "Figures travel as data": tested on the same briefs as 11_retyping. The supervisor no longer
# writes the number. It names which finding answers the question and writes a sentence with a
# slot, and code fills the slot. What that removes, and what it cannot.

import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Literal

sys.path.insert(0, "code")
from openai import OpenAI                                          # noqa: E402
from pydantic import BaseModel                                     # noqa: E402

from clarity.config import MODEL_FAST                              # noqa: E402

client = OpenAI()
random.seed(20)
FIGURES = 300
SUBJECTS = ["revenue", "freight cost", "the discount total", "order value", "gross profit"]
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]


def make_brief():
    """The same draws as 11_retyping, kept as records instead of sentences."""
    findings, seen = [], set()
    while len(findings) < 4:
        subject, region = random.choice(SUBJECTS), random.choice(REGIONS)
        quarter = f"{random.choice([2023, 2024, 2025])} Q{random.randint(1, 4)}"
        if (subject, region, quarter) in seen:
            continue
        seen.add((subject, region, quarter))
        value = round(random.uniform(1_000_000, 9_999_999), 2)
        prior = round(value * random.uniform(0.85, 1.15), 2)
        findings.append({"id": f"F{len(findings) + 1}", "subject": subject, "region": region,
                         "quarter": quarter, "value": value, "value_quarter_before": prior})
    target = random.choice(findings)
    question = (f"What was {target['subject']} for the {target['region']} in {target['quarter']}, "
                "and why did it move?")
    return findings, question, target


class Answer(BaseModel):
    finding_id: str
    field: Literal["value", "value_quarter_before"]
    sentence: str        # must contain {figure} where the number goes


CAUSE = "the documents attribute the movement to an account loss and a supplier price rise"


def structured(findings, question) -> Answer:
    return client.chat.completions.parse(
        model=MODEL_FAST, max_completion_tokens=300, response_format=Answer,
        messages=[{"role": "system", "content":
                   "You are the supervisor. Choose the finding and field that answer the user's "
                   "question, and write the final answer in two or three sentences with the "
                   "placeholder {figure} where the number goes. Never write the number yourself."},
                  {"role": "user", "content": f"Findings: {findings}\nResearcher: {CAUSE}.\n\n"
                                              f"Question: {question}"}]).choices[0].message.parsed


def render(answer: Answer, findings) -> str | None:
    chosen = next((f for f in findings if f["id"] == answer.finding_id),
                  None)
    if chosen is None or "{figure}" not in answer.sentence:
        return None
    figure = f"{chosen[answer.field]:,.2f}"
    return answer.sentence.replace("{figure}", figure)


briefs = [make_brief() for _ in range(FIGURES)]
with ThreadPoolExecutor(max_workers=8) as pool:
    answers = list(pool.map(lambda b: structured(b[0], b[1]), briefs))

tally = {"exact": 0, "the quarter-before field": 0, "a different finding": 0,
         "no slot, or a number written anyway": 0}
example = None
for (findings, _, target), answer in zip(briefs, answers):
    text = render(answer, findings)
    stray = text is not None and len(re.findall(r"\d[\d,]{5,}", answer.sentence)) > 0
    if text is None or stray:
        tally["no slot, or a number written anyway"] += 1
    elif answer.finding_id != target["id"]:
        tally["a different finding"] += 1
        example = example or (target, answer)
    elif answer.field != "value":
        tally["the quarter-before field"] += 1
        example = example or (target, answer)
    else:
        tally["exact"] += 1

print(f"{FIGURES} final answers from the same briefs as 11_retyping,")
print("with the figure filled in by code\n")
for kind, n in tally.items():
    print(f"  {kind:<38} {n:>3}")
if example:
    target, answer = example
    print(f"\n  wanted {target['id']} value; the model chose {answer.finding_id} "
          f"{answer.field}")
print("\nEvery figure that reached an answer is exactly a warehouse value: retyping is")
print("gone by construction. Choosing which value is still the model's job, and its")
print("errors are still possible, now visible as a field that can be checked.")
