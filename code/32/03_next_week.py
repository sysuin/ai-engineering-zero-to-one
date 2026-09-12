# What to build next, decided from the card rather than from opinion.
#
# The run in the previous listing produced 120 rows. This one triages them, which is the
# whole of §32.8: the argument for next week's work is a list of cases, not a feeling.

import json
import statistics as stats
from collections import Counter

card = json.load(open("code/32/_card.json"))
rows = card["rows"]

failed = [r for r in rows if not r["correct"]]
slow = sorted(rows, key=lambda r: -r["seconds"])[:10]
wrongly_refused = [r for r in rows
                   if r["refused"] and r["kind"] != "unanswerable"]

print(f"{len(rows)} runs. Three questions, asked of the same data.\n")

# ---------------------------------------------------------------- 1. what is wrong
print(f"1. What is wrong: {len(failed)} failures, by population.\n")
by_kind = Counter(r["kind"] for r in failed)
population = Counter(r["kind"] for r in rows)
for kind, n in by_kind.most_common():
    print(f"   {kind:<16}{n:>3} of {population[kind]:<4} "
          f"{n / population[kind]:>5.0%} of that population")
if not by_kind:
    print("   none — which is a reason to widen the set, not to celebrate")

# By rate, not by count. The largest population produces the most failures almost by
# definition, and chasing that number means always working on whatever is biggest.
if by_kind:
    worst = max(by_kind, key=lambda k: by_kind[k] / population[k])
    share = by_kind[worst] / population[worst]
    biggest = by_kind.most_common(1)[0][0]
    print(f"\n   Sort by rate, not by count. {biggest} produces the most failures "
          f"because it is")
    print(f"   the largest population; {worst} fails at {share:.0%}, which is the "
          "highest rate and")
    print(f"   therefore the population to work on. It is {population[worst]} cases, "
          "so a fix aimed")
    print("   at it is a fix you can measure — and a fix aimed at 'accuracy' is not.")

# ---------------------------------------------------------------- 2. what is slow
print(f"\n2. What is slow: the ten slowest, against a median of "
      f"{card['p50']:.1f}s.\n")
for r in slow[:5]:
    print(f"   {r['id']:<22}{r['seconds']:>6.1f}s  {r['tokens']:>6,} tokens  "
          f"{r['kind']}")
ratio = slow[0]["seconds"] / card["p50"]
print(f"\n   The slowest is {ratio:.1f}x the median. Tail latency is not an average")
print("   problem: it is a small number of runs doing much more work, and §28.6's")
print("   step budget is the lever that bounds them.")

# ---------------------------------------------------------------- 3. what to add
print(f"\n3. What to add to the eval set.\n")
candidates = {r["id"]: r for r in failed + wrongly_refused + slow[:3]}
overlap = len(failed) + len(wrongly_refused) + 3 - len(candidates)
print(f"   {len(candidates)} cases earned a place: every failure, every wrong refusal,")
print("   and the three slowest. They are already written — that is the point of")
print("   triaging a run rather than inventing cases at a whiteboard.")
print()
print(f"   {len(failed):>3} failures            they are regressions the moment "
      "they are fixed")
print(f"   {len(wrongly_refused):>3} wrong refusals      the most expensive "
      "failure mode to leave in")
print(f"   {len(slow[:3]):>3} slowest runs        a latency budget needs cases "
      "that test it")
print(f"   {overlap:>3} counted twice       a wrong refusal is usually also a "
      "failure")

tokens = [r["tokens"] for r in rows]
spread = max(tokens) / stats.median(tokens)
print(f"\n   And one thing the card does not show: the most expensive answer used "
      f"{spread:.1f}x")
print(f"   the tokens of the median one ({max(tokens):,} against "
      f"{stats.median(tokens):,.0f}). A mean cost per")
print("   answer hides that entirely, which is why §28.1 budgets per request rather")
print("   than per month.")

print("\nThat is the loop. Ship, measure, triage the failures into the set, fix the")
print("worst-rate population, re-measure against the card. Every step of it is")
print("evidence, and none of it needs anybody to have an opinion about what users want.")
