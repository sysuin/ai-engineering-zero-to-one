# timeout: 900
# "Asking a model to reason step by step about which category a support ticket belongs to buys
# you latency and tokens, not accuracy." A prediction, so here it is measured: 150 Meridian
# tickets, 30 from each category, labelled directly and after written reasoning.

import json
import random
import re
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from openai import OpenAI

import sys
sys.path.insert(0, "code")
from clarity.config import MODEL_FAST                  # noqa: E402

client = OpenAI()
CATEGORIES = ["Account", "Billing", "Delivery", "Quality", "Returns"]
tickets = [json.loads(line) for line in open("data/meridian/documents/tickets/tickets.jsonl")]
by_category = defaultdict(list)
for t in tickets:
    by_category[t["category"]].append(t)
rng = random.Random(9)
sample = [t for c in CATEGORIES for t in rng.sample(by_category[c], 30)]
TASK = (f"Classify this support ticket into exactly one category: {', '.join(CATEGORIES)}.\n\n"
        "Ticket: {body}\n\n")
STYLES = {
    "answer directly": "Reply with the category name only.",
    "reason step by step": ("Think step by step about what the customer needs, then give "
                            "the category on a final line as 'Category: <name>'."),
}


def classify(style: str, ticket: dict) -> tuple[str | None, int, float]:
    started = time.perf_counter()
    reply = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=600,
        messages=[{"role": "user", "content": TASK.format(body=ticket["body"]) + STYLES[style]}])
    text = reply.choices[0].message.content or ""
    found = re.findall("|".join(CATEGORIES), text)
    return (found[-1] if found else None), reply.usage.completion_tokens, \
        time.perf_counter() - started


print(f"{len(sample)} tickets, {len(sample) // len(CATEGORIES)} per category\n")
print(f"  {'prompt':<22}{'right':>9}{'output tokens':>15}{'seconds':>9}")
for style in STYLES:
    with ThreadPoolExecutor(max_workers=10) as pool:
        results = list(pool.map(lambda t: classify(style, t), sample))
    right = sum(label == t["category"] for (label, _, _), t in zip(results, sample))
    tokens = statistics.median(r[1] for r in results)
    seconds = statistics.median(r[2] for r in results)
    print(f"  {style:<22}{right:>5}/{len(sample)}{tokens:>15.0f}{seconds:>9.2f}")
print("\n  output tokens and seconds are medians per ticket; the label is the last category named")
