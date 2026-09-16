# When the model is 99% sure, is it right 99% of the time? That property is called
# calibration, and it can be measured with questions whose answers code can check.

import math
import random

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
rng = random.Random(5)
N = 60

rows = []
for _ in range(N):
    a, b = rng.randint(120, 990), rng.randint(12, 99)
    threshold = round(a * b * rng.uniform(0.9, 1.1), -1)          # near the true product
    truth = "Yes" if a * b > threshold else "No"
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, logprobs=True, top_logprobs=5, max_completion_tokens=16,
        messages=[{"role": "user", "content":
                   f"Is {a} × {b} greater than {threshold:.0f}? Do not calculate step by step. "
                   "Answer with exactly one word: Yes or No."}])
    options = {t.token.strip(): math.exp(t.logprob)
               for t in response.choices[0].logprobs.content[0].top_logprobs}
    p_yes, p_no = options.get("Yes", 0.0), options.get("No", 0.0)
    if p_yes + p_no == 0:
        continue
    confidence = max(p_yes, p_no) / (p_yes + p_no)
    answer = "Yes" if p_yes >= p_no else "No"
    rows.append((confidence, answer == truth))

BINS = [(0.5, 0.8), (0.8, 0.95), (0.95, 0.99), (0.99, 1.0001)]
print(f"{len(rows)} questions whose answers Python checked\n")
print(f"  {'confidence':>13} {'questions':>9} {'mean conf.':>10} {'accuracy':>9}")
gap, accuracies = 0.0, []
for lo, hi in BINS:
    group = [(c, ok) for c, ok in rows if lo <= c < hi]
    if not group:
        print(f"  {lo:>5.2f}–{min(hi, 1):<6.2f} {0:>9}")
        continue
    conf = sum(c for c, _ in group) / len(group)
    acc = sum(ok for _, ok in group) / len(group)
    gap += len(group) / len(rows) * abs(conf - acc)
    accuracies.append(acc)
    print(f"  {lo:>5.2f}–{min(hi, 1):<6.2f} {len(group):>9} {conf:>10.3f} {acc:>9.3f}")

overall = sum(ok for _, ok in rows) / len(rows)
mean_conf = sum(c for c, _ in rows) / len(rows)
print(f"\noverall accuracy {overall:.3f}; mean confidence {mean_conf:.3f}")
print(f"expected calibration error {gap:.3f} "
      f"({'over' if mean_conf > overall else 'under'}confident on average)")

rising = all(a <= b for a, b in zip(accuracies, accuracies[1:]))
print("accuracy " + ("rose with every step up in confidence: the number carries information"
                     if rising else "did not rise steadily with confidence"))
print("but a stated 0.9 was " + ("not" if gap > 0.05 else "roughly") + " a 90% chance of being right")
