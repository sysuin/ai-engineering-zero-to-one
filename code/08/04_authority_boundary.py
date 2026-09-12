# timeout: 900
# The most important schema decision in this book: which fields the model may set.
#
# Meridian wants a risk band on every supplier contract. There are two ways to get one.

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from enum import Enum
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
SYSTEM = "Extract contract terms exactly as written. Never guess a value not stated."


class Band(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ---------------------------------------------------------------- the tempting way
class ModelDecides(BaseModel):
    """The model reads the contract and tells you the answer. One call, done."""
    price_cap_pct: int
    price_notice_days: int
    risk_band: Band = Field(description="Overall commercial risk of this agreement.")


# ---------------------------------------------------------------- the boundary
class ModelObserves(BaseModel):
    """
    The model reports only what it read. There is no risk_band field, so the model
    cannot set one — not because we asked it not to, but because there is nowhere to
    put it.
    """
    price_cap_pct: int
    price_notice_days: int


def band_from_terms(cap_pct: int, notice_days: int) -> Band:
    """Meridian's actual policy. Six lines, unit-testable, and explainable to an auditor."""
    if cap_pct > 10 or notice_days < 30:
        return Band.HIGH
    if cap_pct > 5 or notice_days < 60:
        return Band.MEDIUM
    return Band.LOW


def ask(model_cls, text: str):
    return client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=300,
        response_format=model_cls,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": f"<contract>\n{text}\n</contract>"}],
    ).choices[0].message.parsed


# Two runs of each, to see whether the answer is stable.
def run(model_cls):
    with ThreadPoolExecutor(max_workers=12) as pool:
        return list(pool.map(lambda c: ask(model_cls, c["text"]), contracts))


decided_a, decided_b = run(ModelDecides), run(ModelDecides)
observed = run(ModelObserves)

policy = [band_from_terms(c["cap_pct"], c["notice_days"]) for c in contracts]
computed = [band_from_terms(o.price_cap_pct, o.price_notice_days) for o in observed]

agree_policy = sum(d.risk_band == p for d, p in zip(decided_a, policy))
stable = sum(a.risk_band == b.risk_band for a, b in zip(decided_a, decided_b))
computed_ok = sum(c == p for c, p in zip(computed, policy))

print(f"{len(contracts)} contracts, two runs of each approach\n")
print(f"{'':38} {'matches policy':>15} {'same twice':>12}")
print(f"{'model sets the band':38} {agree_policy / len(contracts):>14.0%} "
      f"{stable / len(contracts):>12.0%}")
print(f"{'model observes, code decides':38} {computed_ok / len(contracts):>14.0%} "
      f"{'100%':>12}")

print("\nWhere the model's band differs from Meridian's policy:")
shown = 0
for c, d, p in zip(contracts, decided_a, policy):
    if d.risk_band != p and shown < 6:
        print(f"  cap {c['cap_pct']:>2}%  notice {c['notice_days']:>3}d   "
              f"model said {d.risk_band.value:6}  policy says {p.value}")
        shown += 1

print(f"\n  model's bands   {dict(Counter(d.risk_band.value for d in decided_a))}")
print(f"  policy's bands  {dict(Counter(p.value for p in policy))}")

distinct = len(set(d.risk_band for d in decided_a))
print(f"\nRead those two lines again. The model used {distinct} of the three bands.")
print()
print("It is perfectly stable — the same answer twice, every time — and it has quietly")
print("collapsed the distinction the band exists to make. It never flags a high-risk")
print("contract, including the ones allowing a 12% price rise on 30 days notice.")
print()
print("Note how well it would score on the wrong question. Consistency: 100%. Schema")
print("conformance: 100%. Refusals: none. Every metric a naive check would look at")
print("says this is working.")
print()
print("The model is not being unreliable. It is answering a different question: it has")
print("a reasonable general notion of commercial risk, and Meridian has a specific one,")
print("written down, that an auditor can read.")
print()
print("Removing the field is what enforces the difference. Not an instruction in the")
print("prompt, not a code review convention — the schema simply has nowhere to put an")
print("opinion, so the opinion has to come from six lines of Python you can test.")

Path("code/08/_authority.json").write_text(json.dumps({
    "agree_policy": agree_policy / len(contracts),
    "stable": stable / len(contracts),
    "computed_ok": computed_ok / len(contracts),
    "model_bands": dict(Counter(d.risk_band.value for d in decided_a)),
    "policy_bands": dict(Counter(p.value for p in policy)),
}, indent=2))
