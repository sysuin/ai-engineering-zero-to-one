# timeout: 600
# Where should the model stop and a person start? Give each answer a confidence, then
# choose the threshold that minimises the total cost of review and of mistakes.

import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

CATEGORIES = ["Delivery", "Quality", "Billing", "Returns", "Account"]
rows = [json.loads(line) for line in
        Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()][:200]
client = OpenAI()
PROMPT = ("Classify this support ticket into exactly one category: "
          f"{', '.join(CATEGORIES)}. Reply with the category name only.\n\nTicket: ")

REVIEW_COST = 0.99    # a person triaging one ticket: 1.75 minutes at $34/hour (06/01)
ERROR_COST = 6.00     # an assumption: a misrouted ticket bounces, and someone apologises


def classify(row: dict) -> tuple[str, float, bool]:
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, logprobs=True, top_logprobs=5,
        max_completion_tokens=8,
        messages=[{"role": "user", "content": PROMPT + row["body"]}])
    first = response.choices[0].logprobs.content[0]
    answer = (response.choices[0].message.content or "").strip()
    confidence = math.exp(first.logprob)
    return answer, confidence, answer == row["category"]


with ThreadPoolExecutor(max_workers=8) as pool:
    results = list(pool.map(classify, rows))

print(f"{len(results)} tickets; the model alone is right on "
      f"{sum(ok for _, _, ok in results)}\n")
print(f"  {'threshold':>9} {'automated':>9} {'right when automated':>21} {'cost per 1,000':>15}")

best = None
for threshold in (0.0, 0.5, 0.8, 0.9, 0.99, 0.999, 0.9999, 0.99999, 1.01):
    auto = [(c, ok) for _, c, ok in results if c >= threshold]
    reviewed = len(results) - len(auto)
    wrong = sum(1 for _, ok in auto if not ok)
    cost = (reviewed * REVIEW_COST + wrong * ERROR_COST) / len(results) * 1000
    accuracy = f"{sum(ok for _, ok in auto) / len(auto):.1%}" if auto else "—"
    label = "never" if threshold > 1 else f"{threshold:g}"
    print(f"  {label:>9} {len(auto) / len(results):>9.0%} {accuracy:>21} {'$' + f'{cost:,.0f}':>15}")
    if best is None or cost < best[0]:
        best = (cost, label)

theory = 1 - REVIEW_COST / ERROR_COST
print(f"\ncheapest threshold on these tickets: {best[1]} (${best[0]:,.0f} per 1,000)")
print(f"the textbook rule — automate when P(right) > 1 - review/error = {theory:.2f} — "
      "assumes the confidence is calibrated")
