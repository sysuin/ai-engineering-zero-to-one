# timeout: 1800
# Six ways of answering one question, measured for accuracy, cost and time.
#
# The question needs three facts out of a forty-clause contract and three comparisons
# against a policy. It is exactly the shape of task people reach for clever prompting on.

import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from pydantic import BaseModel, Field

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()

POLICY = ("Meridian's procurement policy requires all three of the following:\n"
          "  1. the price adjustment cap is 8% or less\n"
          "  2. the written notice before a price rise is 60 days or more\n"
          "  3. the payment term is 45 days or less")


def truth(contract: dict) -> bool:
    return (contract["cap_pct"] <= 8 and contract["notice_days"] >= 60
            and contract["payment_days"] <= 45)


TRUTH = [truth(c) for c in contracts]
USAGE: list[tuple[int, int]] = []


def call(messages, **kwargs):
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=kwargs.pop("temperature", 0),
        max_completion_tokens=kwargs.pop("max_completion_tokens", 600),
        messages=messages, **kwargs)
    USAGE.append((response.usage.prompt_tokens, response.usage.completion_tokens))
    return response


def parse(response) -> BaseModel | None:
    USAGE.append((response.usage.prompt_tokens, response.usage.completion_tokens))
    return response.choices[0].message.parsed


def ask(contract: dict, instruction: str, **kwargs) -> str:
    return (call([{"role": "user",
                   "content": f"{POLICY}\n\n<contract>\n{contract['text']}\n</contract>"
                              f"\n\n{instruction}"}], **kwargs)
            .choices[0].message.content or "").strip()


def verdict(text: str) -> bool:
    """Read a yes/no out of whatever came back."""
    lowered = text.lower()
    for marker in ("compliant: yes", "compliant: true", "answer: yes", "yes", "true"):
        if marker in lowered:
            return True
    return False


# ------------------------------------------------------------------ 1. direct
def direct(c):
    return verdict(ask(c, "Is this contract compliant? Reply with YES or NO only.",
                       max_completion_tokens=16))


# ------------------------------------------------------------------ 2. chain of thought
def chain_of_thought(c):
    return verdict(ask(c, "Think step by step, checking each of the three requirements "
                          "in turn. Then end with a final line: 'ANSWER: YES' or "
                          "'ANSWER: NO'."))


# ------------------------------------------------------------------ 3. structured CoT
class Checked(BaseModel):
    cap_pct: int = Field(description="The price adjustment cap, as written.")
    notice_days: int = Field(description="Days of notice before a price rise.")
    payment_days: int = Field(description="Days allowed to pay an invoice.")
    reasoning: str = Field(description="One sentence checking each rule.")
    compliant: bool


def structured_cot(c):
    parsed = parse(client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=600,
        response_format=Checked,
        messages=[{"role": "user",
                   "content": f"{POLICY}\n\n<contract>\n{c['text']}\n</contract>"}]))
    return bool(parsed and parsed.compliant)


# ------------------------------------------------------------------ 4. self-consistency
def self_consistency(c, samples: int = 5):
    votes = [verdict(ask(c, "Is this contract compliant? Reply YES or NO only.",
                         temperature=1.0, max_completion_tokens=16))
             for _ in range(samples)]
    return Counter(votes).most_common(1)[0][0]


# ------------------------------------------------------------------ 5. reflection
def reflection(c):
    draft = ask(c, "Is this contract compliant? Answer in one line, then stop.",
                max_completion_tokens=120)
    critique = call([{"role": "user", "content":
                      f"{POLICY}\n\n<contract>\n{c['text']}\n</contract>\n\n"
                      f"A colleague answered: {draft!r}\n\n"
                      "Check each of the three rules against the contract. Say whether "
                      "the answer is right or wrong, and why."}]
                    ).choices[0].message.content
    final = call([{"role": "user", "content":
                   f"{POLICY}\n\nDraft answer: {draft!r}\nReview: {critique!r}\n\n"
                   "Give the final verdict. Reply YES or NO only."}],
                 max_completion_tokens=16).choices[0].message.content
    return verdict(final or "")


# ------------------------------------------------------------------ 6. decomposition
class Facts(BaseModel):
    """Only what the document says. No verdict field — Chapter 8's boundary."""
    cap_pct: int
    notice_days: int
    payment_days: int


def decomposition(c):
    facts = parse(client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=200,
        response_format=Facts,
        messages=[{"role": "user", "content":
                   "Extract the three figures from this agreement: the price adjustment "
                   "cap percentage, the days of written notice required before a price "
                   "rise, and the days allowed to pay an invoice.\n\n"
                   f"<contract>\n{c['text']}\n</contract>"}]))
    if facts is None:
        return False
    # The decision is made here, in code, and can be unit-tested.
    return (facts.cap_pct <= 8 and facts.notice_days >= 60
            and facts.payment_days <= 45)


PATTERNS = {
    "direct":            direct,
    "chain of thought":  chain_of_thought,
    "structured CoT":    structured_cot,
    "self-consistency":  self_consistency,
    "reflection":        reflection,
    "decomposition":     decomposition,
}

results = {}
print(f"{'pattern':20} {'accuracy':>9} {'calls':>7} {'tokens':>9} {'seconds':>9}")
for name, fn in PATTERNS.items():
    USAGE.clear()
    started = time.time()
    with ThreadPoolExecutor(max_workers=10) as pool:
        answers = list(pool.map(fn, contracts))
    elapsed = time.time() - started

    correct = sum(a == t for a, t in zip(answers, TRUTH))
    tokens = sum(p + c for p, c in USAGE)
    results[name] = {"accuracy": correct / len(contracts), "calls": len(USAGE),
                     "tokens": tokens, "seconds": round(elapsed, 1),
                     "predicted_compliant": sum(answers)}
    print(f"{name:20} {correct / len(contracts):>8.1%} {len(USAGE):>7} "
          f"{tokens:>9,} {elapsed:>9.1f}")

print(f"\n{len(contracts)} contracts, {sum(TRUTH)} of them genuinely compliant.")
print("Predicted compliant, by pattern:")
for name, r in results.items():
    print(f"  {name:20} {r['predicted_compliant']:>3}  (truth: {sum(TRUTH)})")

Path("code/09/_patterns.json").write_text(json.dumps(
    {"truth_compliant": sum(TRUTH), "n": len(contracts), "results": results}, indent=2))
