# timeout: 900
# Reflection scored worst of the six. This is why.

from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()

POLICY = ("Meridian's procurement policy requires all three of the following:\n"
          "  1. the price adjustment cap is 8% or less\n"
          "  2. the written notice before a price rise is 60 days or more\n"
          "  3. the payment term is 45 days or less")


def truth(c):
    return c["cap_pct"] <= 8 and c["notice_days"] >= 60 and c["payment_days"] <= 45


def call(messages, budget=600):
    return (client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=budget,
        messages=messages).choices[0].message.content or "").strip()


def verdict(text: str) -> bool:
    return "yes" in text.lower()[:24] or "true" in text.lower()[:24]


def trace(c):
    body = f"{POLICY}\n\n<contract>\n{c['text']}\n</contract>"
    draft = call([{"role": "user", "content":
                   body + "\n\nIs this contract compliant? Answer in one line."}], 120)
    critique = call([{"role": "user", "content":
                      f"{body}\n\nA colleague answered: {draft!r}\n\nCheck each of the "
                      "three rules against the contract. Say whether the answer is "
                      "right or wrong, and why."}])
    final = call([{"role": "user", "content":
                   f"{POLICY}\n\nDraft answer: {draft!r}\nReview: {critique!r}\n\n"
                   "Give the final verdict. Reply YES or NO only."}], 16)
    return draft, critique, final


with ThreadPoolExecutor(max_workers=10) as pool:
    traces = list(pool.map(trace, contracts))

flipped_right, flipped_wrong, kept = 0, 0, 0
examples = []
for c, (draft, critique, final) in zip(contracts, traces):
    d, f, t = verdict(draft), verdict(final), truth(c)
    if d == f:
        kept += 1
    elif f == t:
        flipped_right += 1
    else:
        flipped_wrong += 1
        if len(examples) < 3:
            examples.append((c, draft, critique, final))

print(f"Of {len(contracts)} contracts, the review step:")
print(f"  left the answer alone      {kept}")
print(f"  changed it from wrong to right   {flipped_right}")
print(f"  changed it from RIGHT TO WRONG   {flipped_wrong}")

print("\nA case where the critic broke a correct answer:")
c, draft, critique, final = examples[0]
print(f"  contract     cap {c['cap_pct']}%, notice {c['notice_days']}d, "
      f"payment {c['payment_days']}d")
print(f"  truth        {'compliant' if truth(c) else 'not compliant'}")
print(f"  draft        {draft[:120]}")
print(f"  critique     {critique[:220].replace(chr(10), ' ')}")
print(f"  final        {final[:40]}")

print()
print("Read that trace carefully, because the failure is not where you would guess.")
print()
print("The draft was correct. The critique AGREED with it — 'the colleague's answer is")
print("right'. And the final step then answered YES, because it was handed two pieces")
print("of prose and asked to reconcile them, and 'the answer is right' looks a great")
print("deal like 'yes'.")
print()
print("No individual step failed. The composition did. Meaning was carried between")
print("steps as English, and English about a verdict is not a verdict.")
print()

# The fix: carry state as data, not as prose.
from pydantic import BaseModel                                       # noqa: E402


class Review(BaseModel):
    draft_was_correct: bool
    corrected_verdict: bool


def typed_chain(c):
    body = f"{POLICY}\n\n<contract>\n{c['text']}\n</contract>"
    draft = call([{"role": "user", "content":
                   body + "\n\nIs this contract compliant? Answer in one line."}], 120)
    parsed = client.chat.completions.parse(
        model=MODEL_FAST, temperature=0, max_completion_tokens=400,
        response_format=Review,
        messages=[{"role": "user", "content":
                   f"{body}\n\nA colleague answered: {draft!r}\n\nCheck each rule. "
                   "Set draft_was_correct, and set corrected_verdict to the verdict you "
                   "believe is right."}],
    ).choices[0].message.parsed
    return bool(parsed and parsed.corrected_verdict)


with ThreadPoolExecutor(max_workers=10) as pool:
    typed = list(pool.map(typed_chain, contracts))
typed_ok = sum(t == truth(c) for t, c in zip(typed, contracts))

prose_ok = sum(verdict(f) == truth(c) for c, (_, _, f) in zip(contracts, traces))
print(f"Same three steps, state carried as prose:      {prose_ok}/{len(contracts)}  "
      f"{prose_ok / len(contracts):.1%}")
print(f"Same three steps, state carried as a boolean:  {typed_ok}/{len(contracts)}  "
      f"{typed_ok / len(contracts):.1%}")
print()
print("Nothing about the reasoning changed. What changed is that the last step reads a")
print("field instead of interpreting a sentence. Chapter 8's schema, used as the join")
print("between two calls rather than as the output of one.")
