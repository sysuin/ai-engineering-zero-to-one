# timeout: 900
# Six wordings of the same instruction, on a task the model does not already get perfect. How
# much does the wording alone move the score — and do the wordings fail on the same tickets?

import json
from concurrent.futures import ThreadPoolExecutor
from itertools import combinations
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
rows = [json.loads(line) for line in
        Path("data/meridian/documents/tickets/tickets.jsonl").read_text().splitlines()][:200]
LABELS = "Delivery, Quality, Billing, Returns, Account"
WORDINGS = {
    "classify into one category": f"Classify this support ticket into exactly one category: {LABELS}. "
                                  "Reply with the category name only.",
    "which queue":                f"Which queue should this ticket go to? The queues are {LABELS}. "
                                  "Answer with the queue name.",
    "label it":                   f"Label the ticket with one of: {LABELS}. Output the label and nothing else.",
    "as a support lead":          f"You triage support tickets. Choose the category — {LABELS} — and "
                                  "reply with just that word.",
    "what is it about":           f"What is this ticket mainly about? Pick from {LABELS}. One word.",
    "the team responsible":       f"Name the team responsible for this ticket: {LABELS}. Reply with the "
                                  "team only.",
}


def classify(instruction: str, row: dict) -> bool:
    answer = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=30,
        messages=[{"role": "user", "content": f"{instruction}\n\nTicket: {row['body']}"}],
    ).choices[0].message.content or ""
    return answer.strip().strip(".").lower() == row["category"].lower()


with ThreadPoolExecutor(max_workers=16) as pool:
    right = {name: list(pool.map(lambda r, w=w: classify(w, r), rows)) for name, w in WORDINGS.items()}

n = len(rows)
print(f"{n} tickets, six wordings of the same instruction\n")
for name, hits in sorted(right.items(), key=lambda kv: -sum(kv[1])):
    print(f"  {name:30} {sum(hits) / n:6.1%}")
scores = [sum(h) / n for h in right.values()]
print(f"\nspread between best and worst wording: {100 * (max(scores) - min(scores)):.1f} points")

always_wrong = sum(not any(right[w][i] for w in right) for i in range(n))
sometimes = sum(0 < sum(right[w][i] for w in right) < len(right) for i in range(n))
print(f"tickets every wording got wrong:        {always_wrong}")
print(f"tickets the wordings disagreed on:      {sometimes}")
agree = [sum(a == b for a, b in zip(right[x], right[y])) / n for x, y in combinations(right, 2)]
print(f"two wordings agree on right/wrong for {min(agree):.0%} to {max(agree):.0%} of tickets")
