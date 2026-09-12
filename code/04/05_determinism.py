# "Set temperature to zero and it becomes deterministic."
#
# Half true, and the half that is false is the half that will catch you out.

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()

SHORT = "Summarise in one sentence: Midwest revenue fell 53% in 2024 Q3."
LONG = ("Write a 120-word note to a sales director explaining why Midwest revenue "
        "fell 53% in 2024 Q3 after the largest account did not renew, and what to "
        "do about it.")


def run(prompt: str, n: int = 6) -> list[str]:
    return [client.chat.completions.create(
                model=MODEL_FAST,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
            ).choices[0].message.content.strip()
            for _ in range(n)]


for label, prompt in (("A short answer", SHORT), ("A long one", LONG)):
    answers = run(prompt)
    lengths = sorted({len(a) for a in answers})
    print(f"{label}, temperature=0, six runs")
    print(f"  distinct answers: {len(set(answers))} of 6")
    print(f"  lengths:          {lengths}")
    print(f"  first 70 chars:   {answers[0][:70]!r}")
    print()

print("temperature=0 means 'always pick the most likely next token'. When one token is")
print("far ahead of the others, that is stable. Over hundreds of tokens, some position")
print("eventually has two near-equal candidates, and floating-point arithmetic on")
print("different hardware breaks the tie differently. The longer the output, the more")
print("chances for that to happen.")
print()
print("So: temperature=0 is a strong nudge, not a contract. Design as though the model")
print("may answer differently tomorrow, because one day it will.")
