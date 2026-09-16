# Text is the data type this book handles most. A handful of moves covers nearly all of it.

import json
import re
from collections import Counter
from pathlib import Path

path = Path("data/meridian/documents/tickets/tickets.jsonl")
# One ticket per line; json.loads turns each line into a dictionary (Chapter 3 explains).
tickets = [json.loads(line) for line in path.read_text().splitlines()]

body = tickets[1]["body"]
print("body:        ", repr(body))
print("len:         ", len(body))
print("upper:       ", body.upper())
print("slice [:6]:  ", repr(body[:6]), "   [-11:]:", repr(body[-11:]))
print("'wash' in:   ", "wash" in body)
print("split:       ", body.split()[:4], "...")
print("join:        ", "_".join(body.split()[:3]))
print("replace:     ", body.replace("cloths", "wipes"))

# Normalising before comparing. Two tickets that "say the same thing" rarely match as-is.
a, b = "  Where is ORDER 25095?? ", "where is order 25095??"
print("\nequal as typed:     ", a == b)
print("equal normalised:   ", a.strip().lower() == b.strip().lower())

# A regular expression matches a PATTERN, not a fixed string. Every SKU code looks like
# MRD-, three capital letters, a dash and three digits.
SKU = re.compile(r"MRD-[A-Z]{3}-\d{3}")
naming = [t for t in tickets if SKU.search(t["body"])]
print(f"\ntickets whose text names a SKU: {len(naming)} of {len(tickets)}")
counts = Counter(code for t in tickets for code in SKU.findall(t["body"]))
print("most named:", counts.most_common(3))

# A group, in brackets, captures just the part you want.
ORDER = re.compile(r"order\s+#?(\d+)", re.IGNORECASE)
orders = [m.group(1) for t in tickets if (m := ORDER.search(t["body"]))]
print(f"tickets quoting an order number: {len(orders)}; first three: {orders[:3]}")

# And the reason to be careful: the text field and the structured field can disagree.
disagree = [t for t in naming if t["sku"] not in SKU.findall(t["body"])]
print(f"tickets naming a SKU in the text other than the one in the sku field: {len(disagree)}")
