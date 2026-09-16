# 120 cases is not 120 independent questions if many are one question with the numbers
# changed. Collapse each question to its template and count what is left.

import re
import sys
from collections import Counter

sys.path.insert(0, "code")
from clarity.evals.runner import load                           # noqa: E402

cases = load()


def template(question: str) -> str:
    q = re.sub(r"\b20\d\d\b", "<year>", question)
    q = re.sub(r"\bQ[1-4]\b", "<quarter>", q)
    q = re.sub(r"MSC-\d{4}-\d{3}", "<contract>", q)
    q = re.sub(r"\b(Northeast|Southeast|Midwest|West|Southwest)\b", "<region>", q)
    return re.sub(r"\d[\d,.]*", "<n>", q)


by_tier: dict[str, Counter] = {}
for case in cases:
    by_tier.setdefault(case["tier"], Counter())[template(case["question"])] += 1

for tier, counts in by_tier.items():
    total = sum(counts.values())
    print(f"{tier}: {total} cases, {len(counts)} distinct templates")
    for text, n in counts.most_common(3):
        print(f"  {n:>3} x {text[:70]}")
    print()

head = by_tier["head"]
largest = head.most_common(1)[0][1]
print(f"The largest template alone is {largest} of {sum(head.values())} head cases. Cases that share")
print("a phrasing tend to pass or fail together — one prompt, one tool, one kind of number —")
print("so they carry less independent evidence than their count suggests, and an interval")
print("computed as if every case were independent, like the one in 02, is too narrow.")
