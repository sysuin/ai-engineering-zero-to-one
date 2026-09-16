# The statistics an eval set needs, from the runs already made: an interval for every
# slice, and a paired test for "did the change help?".
# Reads code/21/_scorecard.json (01) and code/21/_incumbent.json (05).

import json
from math import comb, sqrt

rows = json.load(open("code/21/_scorecard.json"))
pairs = json.load(open("code/21/_incumbent.json"))


def wilson(right: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a proportion that behaves near 0% and 100%, unlike p ± 2 SE."""
    p = right / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    half = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / (1 + z * z / total)
    return max(0.0, centre - half), min(1.0, centre + half)


print("every slice, with its 95% interval\n")
kinds: dict[str, list[int]] = {}
for row in rows:
    kinds.setdefault(row["kind"], []).append(int(row["correct"]))
kinds["overall"] = [int(r["correct"]) for r in rows]
for kind, marks in kinds.items():
    low, high = wilson(sum(marks), len(marks))
    print(f"  {kind:<14} {sum(marks):>3}/{len(marks):<4} {sum(marks) / len(marks):>4.0%}"
          f"   {low:>4.0%} to {high:.0%}")

# Paired: the same cases, before and after. Only the cases that changed carry evidence.
before, after = pairs["incumbent"], pairs["current"]
fixed = sum(1 for k in before if not before[k] and after[k])
broke = sum(1 for k in before if before[k] and not after[k])
n = fixed + broke
# Exact McNemar test: if the change did nothing, each changed case is a coin toss.
p_value = min(1.0, 2 * sum(comb(n, i) for i in range(min(fixed, broke) + 1)) / 2 ** n)
print(f"\npaired comparison, v0.5 against v0.9 on the same {len(before)} cases")
print(f"  unchanged cases carry no evidence about the change: {len(before) - n}")
print(f"  fixed {fixed}, broke {broke}; exact McNemar p = {p_value:.2g}")

# The same test on a small change: what a real five-point improvement looks like.
for small_fixed, small_broke in ((9, 3), (7, 1)):
    m = small_fixed + small_broke
    p = min(1.0, 2 * sum(comb(m, i) for i in range(small_broke + 1)) / 2 ** m)
    print(f"  a change that fixed {small_fixed} and broke {small_broke}: net "
          f"{small_fixed - small_broke} of {len(before)}, p = {p:.2f}")

slices = len(kinds) - 1
print(f"\nread {slices} independent slices at the 5% level and at least one will look")
print(f"significant about {1 - 0.95 ** slices:.0%} of the time by chance alone")
