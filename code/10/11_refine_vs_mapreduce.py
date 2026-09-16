# timeout: 2400
# "Refine loses the beginning": tested. A long document with one notable fact in each of 24
# parts, summarised by map-reduce and by refine to the same length, then questioned from the
# summary alone. Which facts survive, by where they were in the document?

import random
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
APPENDIX = Path("data/meridian/documents/awkward/03-long-appendix.md").read_text()
SECTIONS = ["## " + s for s in APPENDIX.split("## ") if s.strip()][1:]
PARTS, PER_PART, WORDS, RUNS = 24, 6, 300, 3
SUPPLIERS = ["Ashford", "Brennan", "Calloway", "Dunmore", "Everly", "Fairholm", "Garside",
             "Hollins", "Ingram", "Jessop", "Kettering", "Lindqvist", "Marlowe", "Norcott",
             "Oakley", "Pendry", "Quayle", "Redfern", "Stanwick", "Thackeray", "Upton",
             "Vance", "Whitlock", "Yardley"]


def call(prompt: str, budget: int) -> str:
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=budget,
        messages=[{"role": "user", "content": prompt}]).choices[0].message.content or "").strip()


def build(seed: int) -> tuple[list[str], list[tuple[int, str]]]:
    rng = random.Random(seed)
    names = rng.sample(SUPPLIERS, PARTS)
    parts, facts = [], []
    for p in range(PARTS):
        sections = SECTIONS[p * PER_PART:(p + 1) * PER_PART]
        item = 1 + p * PER_PART + rng.randrange(PER_PART)
        notice = (f"## Notice for item C.{item}\n\nItem C.{item} was withdrawn this year, and its "
                  f"replacement is now sourced from {names[p]} Industrial.\n")
        sections.insert(rng.randrange(len(sections) + 1), notice)
        parts.append("\n".join(sections))
        facts.append((item, names[p]))
    return parts, facts


def map_reduce(parts: list[str]) -> str:
    with ThreadPoolExecutor(max_workers=8) as pool:
        notes = list(pool.map(lambda text: call(
            f"Summarise this part of a specification appendix in at most 60 words.\n\n{text}", 150),
            parts))
    return call(f"Combine these notes on a specification appendix into one summary of at most {WORDS} "
                "words.\n\n" + "\n\n".join(notes), 700)


def refine(parts: list[str]) -> str:
    summary = ""
    for text in parts:
        summary = call(f"Here is a running summary of a specification appendix, and the next part of "
                       f"it. Update the summary to cover both, in at most {WORDS} words.\n\n"
                       f"<summary>\n{summary or '(empty)'}\n</summary>\n\n<part>\n{text}\n</part>", 700)
    return summary


def score(summary: str, facts: list[tuple[int, str]]) -> list[bool]:
    def ask(fact):
        item, name = fact
        answer = call(f"<notes>\n{summary}\n</notes>\n\nUsing only these notes: who now supplies the "
                      f"replacement for item C.{item}? Reply with the name, or NOT STATED.", 20)
        return name.lower() in answer.lower()
    with ThreadPoolExecutor(max_workers=8) as pool:
        return list(pool.map(ask, facts))


third = PARTS // 3
print(f"{PARTS} parts, one notice in each naming a new supplier; both summaries limited to "
      f"{WORDS} words; {RUNS} runs\n")
print(f"  {'method':12}{'first third':>13}{'middle':>9}{'last third':>12}{'all':>9}{'words':>7}")
totals = {}
for name, method in (("map-reduce", map_reduce), ("refine", refine)):
    hits, words = [[], [], []], []
    for run in range(RUNS):
        parts, facts = build(10 + run)
        summary = method(parts)
        words.append(len(summary.split()))
        found = score(summary, facts)
        for t in range(3):
            hits[t] += found[t * third:(t + 1) * third]
    cells = "".join(f"{sum(h):>{w - 3}}/{len(h):<2}" for h, w in zip(hits, (13, 9, 12)))
    everything = sum(map(sum, hits))
    print(f"  {name:12}{cells}{everything:>5}/{PARTS * RUNS}{sum(words) / RUNS:>7.0f}")
