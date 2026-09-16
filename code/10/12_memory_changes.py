# timeout: 2400
# Memory when facts change. Eight facts agreed at the start, two of them changed later, and a
# summary and a typed state each updated every ten turns as a real system would. At the end:
# which strategy still knows the current value, and which reports the old one?

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
RUNS, EVERY = 8, 10

EARLY = [
    ("We're preparing the pack for Halloway Group, the Midwest account.", "Understood: Halloway Group, Midwest."),
    ("Use 2024 Q2 as the comparison quarter.", "Noted: comparisons against 2024 Q2."),
    ("Report all money in thousands of dollars.", "Will do: thousands of dollars."),
    ("It's for the regional director, so keep it short.", "Short, for the regional director."),
    ("They need it by Friday.", "Due Friday."),
    ("Use bar charts throughout, no pie charts.", "Bar charts only."),
    ("Leave the Packaging category out entirely.", "Packaging excluded."),
]
# Forty turns of ordinary work, some about other accounts, quarters and units: the kind of
# conversation in which a summary has to choose what to keep. The other quarters and units are
# never this pack's old or new values, so a stale answer can be told from a confused one.
REGIONS = ["Northeast", "Southeast", "Southwest", "West"]
OTHERS = ["Voss Industrial", "Thorne Chemical", "Bexley Foods", "Carrow Health"]
FILLER = []
for n in range(40):
    year, quarter = (2023, 2025)[n % 2], ("Q3", "Q4")[(n // 2) % 2]
    if n % 4 == 3:
        other = OTHERS[(n // 4) % 4]
        FILLER.append((f"Separately, the {other} pack next month will compare against {year} {quarter}, "
                       "in euros.", f"Noted for the {other} pack."))
    else:
        FILLER.append((f"What was {REGIONS[n % 4]} revenue in {year} {quarter}?",
                       f"About ${1.2 + 0.037 * n:.1f}m."))
CHANGES = {14: ("Actually, switch the comparison quarter to 2024 Q1.", "Changed: comparisons against 2024 Q1."),
           26: ("Change of plan on money: show it in millions of dollars.", "Changed: millions of dollars.")}
pairs = list(EARLY)
for n, filler in enumerate(FILLER):
    pairs.append(CHANGES.get(len(pairs), filler))
FINAL = ("Before I send the Halloway Group pack, confirm its brief in one line each: region, "
         "comparison quarter, money unit, audience, deadline, chart style, excluded category.")

# (fact, what the current value must contain, what it must not)
CHECKS = [("region", r"midwest", None),
          ("quarter", r"q1", r"q2"), ("unit", r"million", r"thousand"),
          ("audience", r"director", None), ("deadline", r"friday", None),
          ("charts", r"\bbar", None), ("excluded", r"packaging", None)]


def turns(ps):
    return [m for u, a in ps for m in ({"role": "user", "content": u},
                                       {"role": "assistant", "content": a})]


class Brief(BaseModel):
    account: str | None
    region: str | None
    comparison_quarter: str | None
    money_unit: str | None
    audience: str | None
    deadline: str | None
    chart_style: str | None
    excluded_category: str | None


class Briefs(BaseModel):
    """One brief per pack being discussed, so another account's details have somewhere to go."""
    packs: list[Brief]


def rolling(kind: str, older: list, sentences: str = "three sentences") -> str:
    """Fold the older turns in ten at a time, each update seeing the last result and new turns."""
    memory = ""
    for i in range(0, len(older), 2 * EVERY):
        chunk = json.dumps(older[i:i + 2 * EVERY])
        if kind == "summary":
            memory = client.chat.completions.create(
                model=MODEL_FAST, temperature=0, max_completion_tokens=150,
                messages=[{"role": "user", "content": "Update this summary of a conversation with the new "
                           f"turns, in at most {sentences}.\n\nSummary: {memory or '(none)'}\n\n"
                           f"New turns: {chunk}"}]).choices[0].message.content or ""
        else:
            shape = Briefs if kind == "per pack" else Brief
            memory = client.chat.completions.parse(
                model=MODEL_FAST, temperature=0, max_completion_tokens=600, response_format=shape,
                messages=[{"role": "user", "content": "Update what this conversation has established, "
                           "replacing any value the new turns change.\n\n"
                           f"Established: {memory or '{}'}\n\nNew turns: {chunk}"}]
            ).choices[0].message.parsed.model_dump_json()
    return memory


history = turns(pairs)
recent = history[-2 * EVERY:]
older = history[:-2 * EVERY]


def messages_for(name: str) -> list:
    if name == "keep everything":
        return [SYSTEM] + history
    if name == "last 10 turns":
        return [SYSTEM] + recent
    if "state" in name:
        kind = "per pack" if "per pack" in name else "state"
        return [SYSTEM, {"role": "system", "content": "Established: " + rolling(kind, older)}] + recent
    sentences = "one sentence" if "1 sentence" in name else "three sentences"
    return [SYSTEM, {"role": "system", "content": "Earlier: " + rolling("summary", older, sentences)}] + recent


def run(name: str) -> tuple[dict, int, str]:
    messages = messages_for(name) + [{"role": "user", "content": FINAL}]
    memory = messages[1]["content"] if messages[1]["role"] == "system" and len(messages) > 2 else ""
    answer = (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200, messages=messages,
    ).choices[0].message.content or "").lower()
    lines = {fact: next((l for l in answer.splitlines() if key in l), "")
             for fact, key in (("region", "region"), ("quarter", "quarter"),
                               ("unit", "unit"), ("audience", "audience"), ("deadline", "deadline"),
                               ("charts", "chart"), ("excluded", "exclud"))}
    marks = {}
    for fact, want, stale in CHECKS:
        line = lines[fact] or answer
        if stale and re.search(stale, line) and not re.search(want, line):
            marks[fact] = "old"
        else:
            marks[fact] = "ok" if re.search(want, line) else "lost"
    return marks, sum(len(encoder.encode(m["content"])) for m in messages), memory


NAMES = ["keep everything", "last 10 turns", "rolling summary + last 10",
         "1 sentence summary + last 10", "rolling state + last 10", "state per pack + last 10"]
print(f"{len(pairs)} turns: 8 facts agreed in the first 7, the quarter changed at turn 15 and the")
print("money unit at turn 27, other accounts' quarters and currencies discussed throughout;")
print(f"summary and state updated every {EVERY} turns; {RUNS} runs each\n")
print(f"  {'strategy':29}{'tokens':>6}{'right':>7}{'old':>6}{'lost':>6}   changed facts right")
tally = {}
for name in NAMES:
    with ThreadPoolExecutor(max_workers=RUNS) as pool:
        results = list(pool.map(lambda _: run(name), range(RUNS)))
    marks = [m for m, _, _ in results]
    count = lambda v: sum(list(m.values()).count(v) for m in marks)  # noqa: E731
    wrong = [memory for m, _, memory in results if any(v != "ok" for v in m.values())]
    tally[name] = (count("ok"), len(wrong), wrong[0] if wrong else "")
    changed = " ".join(f"{f} {sum(m[f] == 'ok' for m in marks)}/{RUNS}" for f in ("quarter", "unit"))
    print(f"  {name:29}{results[0][1]:>6,}{count('ok'):>4}/{len(CHECKS) * RUNS}{count('old'):>6}"
          f"{count('lost'):>6}   {changed}")

print()
for name in NAMES[2:]:
    print(f"  {name}: {tally[name][1]} of {RUNS} runs got at least one fact wrong")
for name in NAMES[4:]:
    if tally[name][2]:
        print(f"\nwhat '{name}' carried in a run that went wrong:")
        carried = tally[name][2][:600]
        for i in range(0, len(carried), 76):
            print("  " + carried[i:i + 76])
