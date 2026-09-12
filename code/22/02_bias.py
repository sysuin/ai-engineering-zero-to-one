# timeout: 2400
# Reads code/22/_answerset.json. Three biases, measured on real judgements.

import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.judge import (PAIRWISE,                     # noqa: E402
                                 PAIRWISE_REFERENCE, Judge)

rows = json.load(open("code/22/_answerset.json"))
by_id: dict[str, dict[str, dict]] = {}
for row in rows:
    by_id.setdefault(row["id"], {})[row["variant"]] = row
cases = [v for v in by_id.values() if {"plain", "wrong", "padded", "terse"} <= set(v)]
judge = Judge(system=PAIRWISE)
anchored = Judge(system=PAIRWISE_REFERENCE)

# --------------------------------------------------------------- position
# The same two answers, twice, with the sides swapped. A judge with no position bias
# gives the same winner both times.
def both_orders(pair: tuple[dict, dict]) -> tuple[str, str]:
    good, bad = pair
    first = judge.compare(good["question"], good["answer"], bad["answer"])
    second = judge.compare(good["question"], bad["answer"], good["answer"])
    return first, second


pairs = [(c["plain"], c["wrong"]) for c in cases]
with ThreadPoolExecutor(max_workers=8) as pool:
    orders = list(pool.map(both_orders, pairs))

# "correct wins" means A when the correct answer was A, B when it was B.
right_first = sum(1 for a, _ in orders if a == "A")
right_second = sum(1 for _, b in orders if b == "B")
consistent = sum(1 for a, b in orders
                 if (a == "A" and b == "B") or (a == "B" and b == "A"))
picked_a = sum(1 for a, b in orders for v in (a, b) if v == "A")
print(f"POSITION — {len(pairs)} pairs, each judged twice with the sides swapped\n")
print(f"  correct answer in slot A, judge picked it   {right_first} of {len(pairs)}")
print(f"  correct answer in slot B, judge picked it   {right_second} of {len(pairs)}")
print(f"  judge picked slot A, either way             {picked_a} of {2 * len(pairs)}")
print(f"  gave the same verdict both ways             {consistent} of {len(pairs)}")

# --------------------------------------------------------------- verbosity
# Same fact, same correctness, different length. Any winner here is a length preference.
def longer_wins(case: dict) -> str:
    return judge.compare(case["padded"]["question"], case["padded"]["answer"],
                         case["terse"]["answer"])


with ThreadPoolExecutor(max_workers=8) as pool:
    verbosity = Counter(pool.map(longer_wins, cases))
print(f"\nVERBOSITY — the same correct fact, padded with 60 true but irrelevant words,")
print(f"against the bare fact. Both are right, so a tie is the only correct answer.\n")
print(f"  the padded answer won   {verbosity['A']} of {len(cases)}")
print(f"  the bare answer won     {verbosity['B']} of {len(cases)}")
print(f"  tie                     {verbosity['TIE']} of {len(cases)}")

# --------------------------------------------------------------- the mitigation
# Run both orders and only accept a verdict the judge gives twice.
def consensus(pair: tuple[dict, dict]) -> str:
    good, bad = pair
    first = judge.compare(good["question"], good["answer"], bad["answer"])
    second = judge.compare(good["question"], bad["answer"], good["answer"])
    if first == "A" and second == "B":
        return "correct"
    if first == "B" and second == "A":
        return "wrong"
    return "undecided"


settled = Counter(c for c in (
    "correct" if (a == "A" and b == "B") else
    "wrong" if (a == "B" and b == "A") else "undecided"
    for a, b in orders))
print(f"\nMITIGATION — take a verdict only when both orders agree\n")
print(f"  picked the correct answer   {settled['correct']} of {len(pairs)}")
print(f"  picked the wrong answer     {settled['wrong']} of {len(pairs)}")
print(f"  refused to decide           {settled['undecided']} of {len(pairs)}")

decided = settled["correct"] + settled["wrong"]
accuracy = settled["correct"] / decided if decided else 0
naive = right_first / len(pairs)


# --------------------------------------------------------------- with a reference
# The judge above was comparing two answers about a company it knows nothing about.
# Give it the expected answer and ask again, both ways.
def anchored_orders(pair: tuple[dict, dict]) -> tuple[str, str]:
    good, bad = pair
    expected = good["expected"]
    return (anchored.compare(good["question"], good["answer"], bad["answer"],
                             expected),
            anchored.compare(good["question"], bad["answer"], good["answer"],
                             expected))


with ThreadPoolExecutor(max_workers=8) as pool:
    ref_orders = list(pool.map(anchored_orders, pairs))
ref_a = sum(1 for a, b in ref_orders for v in (a, b) if v == "A")
ref_consistent = sum(1 for a, b in ref_orders
                     if (a == "A" and b == "B") or (a == "B" and b == "A"))
ref_right = sum(1 for a, _ in ref_orders if a == "A")
print(f"\nWITH A REFERENCE — the same pairs, the same two orders, plus the expected "
      f"answer\n")
print(f"  correct answer in slot A, judge picked it   {ref_right} of {len(pairs)}")
print(f"  judge picked slot A, either way             {ref_a} of {2 * len(pairs)}")
print(f"  gave the same verdict both ways             {ref_consistent} of "
      f"{len(pairs)}")

print()
print(f"Read the two accuracies against each other. Taking the first verdict as given")
print(f"is right {naive:.0%} of the time. Requiring both orders to agree is right")
print(f"{accuracy:.0%} of the time on the {decided} it will commit to, and admits it "
      f"cannot tell on")
print(f"the other {settled['undecided']}.")
print()
print("The second is a better instrument and a worse product. It costs twice as much,")
print("it returns nothing on some fraction of cases, and that fraction is the honest")
print("output — those are the pairs where the judge was going to be a coin toss and")
print("you would never have known.")
print()
print("The last block is the important one. The same judge, the same pairs, the")
print("same two orders — but told what the answer should be — picked slot A")
print(f"{ref_a} times out of {2 * len(pairs)} and agreed with itself {ref_consistent} "
      f"times out of {len(pairs)}.")
print()
print("So most of what looks like position bias here is not really a preference for")
print("the left-hand side. It is what a judge does when it has no way to tell the")
print("answers apart: something has to be returned, and slot A is what comes out.")
print()
print("Both readings are worth keeping. Bias is real and measurable — present the")
print("same content two ways and count the flips — and the first thing to try is not")
print("a debiasing trick but giving the judge the information it was missing.")

json.dump({"orders": orders, "verbosity": dict(verbosity),
           "settled": dict(settled), "n": len(pairs),
           "ref_orders": ref_orders, "ref_a": ref_a,
           "ref_consistent": ref_consistent, "ref_right": ref_right},
          open("code/22/_bias.json", "w"), indent=1)
