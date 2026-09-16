# timeout: 900
# Measuring a list extraction: every period in days, with its purpose. Precision asks how
# many extracted items are real; recall asks how many real items were extracted.

import re
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

from openai import OpenAI
from pydantic import BaseModel

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()


class Purpose(str, Enum):
    PRICE_NOTICE = "price_notice"
    PAYMENT = "payment"
    REJECTION_WINDOW = "rejection_window"
    TERMINATION_NOTICE = "termination_notice"
    BREACH_REMEDY = "breach_remedy"


class Period(BaseModel):
    clause: str
    days: int
    purpose: Purpose


class Periods(BaseModel):
    periods: list[Period]


CLAUSE_PURPOSE = {"2": "price_notice", "3": "payment", "5": "rejection_window"}


def truth(text: str) -> set[tuple[str, int, str]]:
    items = set()
    for m in re.finditer(r"(\d+) days", text):
        clause = re.findall(r"^(\d+\.\d+)", text[:m.start()], re.M)[-1]
        section = clause.split(".")[0]
        purpose = CLAUSE_PURPOSE.get(section) or ("termination_notice" if clause == "7.1"
                                                 else "breach_remedy")
        items.add((clause, int(m.group(1)), purpose))
    return items


def extract(text: str) -> set[tuple[str, int, str]]:
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, response_format=Periods, max_completion_tokens=600,
        messages=[{"role": "user", "content": "List every period stated in days in this "
                   "agreement, with its clause number and purpose.\n\n<contract>\n"
                   f"{text}\n</contract>"}],
    ).choices[0].message.parsed
    return {(p.clause.strip(), p.days, p.purpose.value) for p in parsed.periods}


with ThreadPoolExecutor(max_workers=12) as pool:
    found = list(pool.map(lambda c: extract(c["text"]), contracts))
real = [truth(c["text"]) for c in contracts]


def scores(project):
    tp = sum(len({project(x) for x in f} & {project(x) for x in r}) for f, r in zip(found, real))
    n_found = sum(len({project(x) for x in f}) for f in found)
    n_real = sum(len({project(x) for x in r}) for r in real)
    p, rc = tp / n_found, tp / n_real
    return p, rc, 2 * p * rc / (p + rc)


print(f"{len(contracts)} contracts, {sum(len(r) for r in real)} real periods\n")
print(f"  {'what has to match':30} {'precision':>9} {'recall':>7} {'F1':>6}")
for label, project in [("days only", lambda x: x[1]),
                       ("days and purpose", lambda x: (x[1], x[2])),
                       ("clause, days and purpose", lambda x: x)]:
    p, r, f1 = scores(project)
    print(f"  {label:30} {p:>9.1%} {r:>7.1%} {f1:>6.1%}")

from collections import Counter  # noqa: E402

extras = Counter((x[2], x[1]) for f, r in zip(found, real) for x in f - r)
missing = Counter((x[2], x[1]) for f, r in zip(found, real) for x in r - f)
print(f"\nextracted but not real: {sum(extras.values())}; most common: {extras.most_common(3)}")
print(f"real but not extracted: {sum(missing.values())}")
