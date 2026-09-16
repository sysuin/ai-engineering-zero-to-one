# timeout: 900
# Retrieval returned the wrong quarter's review, and nothing else. Does the answer notice? Each
# quarter's question is asked with only the previous quarter's section as the excerpt: once the
# Summary, which names its quarter in the text, and once the Commentary, which does not — each with
# and without the source label naming the file.

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                  # noqa: E402
from clarity.evals.runner import plain                 # noqa: E402
from clarity.prompts import load as load_prompt        # noqa: E402

client = OpenAI()
RUNS = 2
REVIEWS = sorted(Path("data/meridian/documents/quarterly-reviews").glob("qbr-*.md"))
CAUGHT = re.compile(r"\bnot\b|\bno\b|n't\b|cannot|only (?:cover|refer|give|provide|show|state|say)"
                    r"|another quarter|different quarter|other quarter|instead of", re.I)


def section(path: Path, heading: str) -> str:
    text = path.read_text()
    return text.split(f"### {heading}")[1].split("###")[0].strip()


def period(path: Path) -> str:
    return path.stem.replace("qbr-", "").replace("-", " ")


def ask(label: str | None, heading: str, text: str, question: str) -> str:
    excerpt = f"[{label}] {heading}\n{text}" if label else f"{heading}\n{text}"
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        messages=[{"role": "system", "content": load_prompt("answer")},
                  {"role": "user", "content": f"<excerpts>\n{excerpt}\n</excerpts>\n\n"
                                              f"Question: {question}"}])
    return reply.choices[0].message.content or ""


jobs = []
for earlier, asked in zip(REVIEWS, REVIEWS[1:]):
    summary, commentary = section(earlier, "Summary"), section(earlier, "Commentary")
    revenue = re.search(r"was (\$[\d,]+)", summary).group(1)
    weakest = re.search(r"(\w+) was the weakest", commentary).group(1)
    for labelled in (True, False):
        label = earlier.name if labelled else None
        jobs.append(("Summary", labelled, label, summary, revenue,
                     f"What was revenue in {period(asked)}?"))
        jobs.append(("Commentary", labelled, label, commentary, weakest,
                     f"Which region was weakest in {period(asked)}?"))
jobs = [job for job in jobs for _ in range(RUNS)]

with ThreadPoolExecutor(max_workers=10) as pool:
    answers = list(pool.map(lambda j: ask(j[2], j[0], j[3], j[5]), jobs))


def answered_from_it(answer: str, value: str) -> bool:
    """The first sentence states the earlier quarter's value, and does not flag a problem."""
    first = re.split(r"(?<=[.!?])\s+", answer.strip(), maxsplit=1)[0]
    return value.lower() in plain(first) and not CAUGHT.search(plain(first))


pairs = len(REVIEWS) - 1
print(f"{pairs} quarters asked about, each given only the previous quarter's section; "
      f"{RUNS} runs each\n")
print(f"  {'excerpt':<44}{'answered from it':>18}")
saved = []
for heading, note in (("Summary", "names its quarter"), ("Commentary", "does not")):
    for labelled in (True, False):
        rows = [(j, a) for j, a in zip(jobs, answers) if j[0] == heading and j[1] == labelled]
        wrong = sum(answered_from_it(a, j[4]) for j, a in rows)
        name = f"{heading} ({note}), {'with' if labelled else 'no'} label"
        print(f"  {name:<44}{wrong:>14}/{len(rows)}")
        saved += [{"excerpt": name, "question": j[5], "value": j[4], "answer": a,
                   "answered_from_it": answered_from_it(a, j[4])} for j, a in rows]
print("\n  answered from it = the first sentence gave the earlier quarter's figure or region as")
print("  the answer, without saying the excerpt was for another period")
json.dump(saved, open("code/13/_wrong_quarter.json", "w"), indent=2)
