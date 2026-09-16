# timeout: 1800
# Four ways to remember a long conversation, tested on the one thing that matters: can the
# system still use something agreed at the start, forty turns later?

import json
import re
from concurrent.futures import ThreadPoolExecutor

import tiktoken
from openai import OpenAI
from pydantic import BaseModel

from clarity.config import MODEL_FAST

client = OpenAI()
encoder = tiktoken.encoding_for_model(MODEL_FAST)
SYSTEM = {"role": "system", "content": "You are Clarity, an analyst's assistant. Be brief."}

# The early turns establish three facts. Forty turns of routine chatter follow.
EARLY = [
    ("We're preparing the pack for Halloway Group, the Midwest account.",
     "Understood — Halloway Group, Midwest."),
    ("Use 2024 Q2 as the comparison quarter, not Q3.",
     "Noted: comparisons against 2024 Q2."),
    ("Report all money in thousands of dollars, rounded.",
     "Will do: thousands, rounded."),
]
FILLER = [(f"Quick one: what does the abbreviation SKU {n} stand for in general?",
           "SKU means stock keeping unit.") for n in range(40)]
FINAL = ("Write the one-line heading for the comparison slide: the account, the comparison "
         "quarter, and the unit money is shown in.")


def turns(pairs):
    out = []
    for user, assistant in pairs:
        out += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
    return out


history = turns(EARLY + FILLER)


class State(BaseModel):
    account: str | None
    comparison_quarter: str | None
    money_unit: str | None


def summary_of(messages) -> str:
    return client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=150,
        messages=[{"role": "user", "content": "Summarise this conversation in three sentences.\n\n"
                   + json.dumps(messages)}]).choices[0].message.content


def state_of(messages) -> State:
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=120, response_format=State,
        messages=[{"role": "user", "content": "Record what this conversation has established.\n\n"
                   + json.dumps(messages)}]).choices[0].message.parsed


STRATEGIES = {
    "keep everything":        lambda: [SYSTEM] + history,
    "last 10 turns":          lambda: [SYSTEM] + history[-20:],
    "summary + last 10":      lambda: [SYSTEM, {"role": "system", "content":
                                        "Earlier: " + summary_of(history[:-20])}] + history[-20:],
    "typed state + last 10":  lambda: [SYSTEM, {"role": "system", "content":
                                        "Established: " + state_of(history[:-20]).model_dump_json()}]
                                       + history[-20:],
}


def check(answer: str) -> dict[str, bool]:
    a = answer.lower()
    return {"account": "halloway" in a, "quarter": "q2" in a,
            "unit": bool(re.search(r"thousand|000s|\$k|\(k\)|\bk\b", a))}


examples = {}
print(f"{len(history) // 2} turns of history; the facts that matter are in the first 3\n")
print(f"  {'strategy':24} {'tokens':>6} {'account':>8} {'quarter':>8} {'unit':>5}")
for name, build in STRATEGIES.items():
    messages = build() + [{"role": "user", "content": FINAL}]
    tokens = sum(len(encoder.encode(m["content"])) for m in messages)
    with ThreadPoolExecutor(max_workers=3) as pool:
        answers = list(pool.map(lambda _: client.chat.completions.create(
            model=MODEL_FAST, temperature=0, max_completion_tokens=60, messages=messages,
        ).choices[0].message.content or "", range(3)))
    marks = [check(a) for a in answers]
    counts = {k: sum(m[k] for m in marks) for k in ("account", "quarter", "unit")}
    print(f"  {name:24} {tokens:>6,} {counts['account']:>6}/3 {counts['quarter']:>6}/3 "
          f"{counts['unit']:>3}/3")
    examples[name] = answers[0].strip()

print()
for name, heading in examples.items():
    print(f"  {name:24} {heading[:62]!r}")
