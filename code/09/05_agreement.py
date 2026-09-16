# timeout: 900
# Before paying for self-consistency, measure the one thing it depends on: when the model
# is wrong, do its samples disagree?

from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

from _contracts import load
from clarity.config import MODEL_FAST

client = OpenAI()
contracts = load()
SAMPLES = 5
POLICY = ("Meridian's procurement policy requires all three of the following:\n"
          "  1. the price adjustment cap is 8% or less\n"
          "  2. the written notice before a price rise is 60 days or more\n"
          "  3. the payment term is 45 days or less")


def truth(c):
    return c["cap_pct"] <= 8 and c["notice_days"] >= 60 and c["payment_days"] <= 45


def sample(c) -> list[bool | None]:
    out = []
    for _ in range(SAMPLES):
        text = (client.chat.completions.create(
            model=MODEL_FAST, temperature=1.0, max_completion_tokens=16,
            messages=[{"role": "user", "content": f"{POLICY}\n\n<contract>\n{c['text']}\n"
                       "</contract>\n\nIs this contract compliant? Reply YES or NO only."}],
        ).choices[0].message.content or "").strip().upper()
        out.append(True if text.startswith("YES") else False if text.startswith("NO") else None)
    return out


with ThreadPoolExecutor(max_workers=10) as pool:
    votes = list(pool.map(sample, contracts))

buckets = Counter()
for c, v in zip(contracts, votes):
    right = sum(x == truth(c) for x in v)
    buckets[right] += 1

print(f"{len(contracts)} contracts, {SAMPLES} samples each at temperature 1.0\n")
print(f"  {'samples right':>13} {'contracts':>10}")
for k in range(SAMPLES, -1, -1):
    print(f"  {k:>9} of {SAMPLES} {buckets[k]:>10}")

unanimous_wrong = buckets[0]
split = sum(buckets[k] for k in range(1, SAMPLES))
print(f"\ncontracts where every sample was wrong: {unanimous_wrong}")
print(f"contracts where the samples disagreed:  {split}")
print("Voting can only rescue a disagreeing contract, and only when most samples are right.")
