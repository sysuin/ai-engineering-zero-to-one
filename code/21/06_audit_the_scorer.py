# The scorer is code, and code can be wrong. Audit it against the answers it scored:
# where does "did it refuse?" disagree with "is the answer in the reply?"
# Reads code/21/_scorecard.json, written by 01.

import json
import re
import sys

sys.path.insert(0, "code")
from clarity.evals.runner import REFUSALS, abstained, load, normalise, plain  # noqa: E402

cases = {c["id"]: c for c in load()}
rows = json.load(open("code/21/_scorecard.json"))


def contains_gold(case: dict, answer: str) -> bool:
    wanted = [case["answer"]] + list(case.get("accept") or [])
    return any(w and normalise(w) in normalise(answer) for w in wanted)


def triggers(answer: str) -> list[str]:
    return [p for p in REFUSALS if p in plain(answer)]


answerable = [r for r in rows if cases[r["id"]]["kind"] != "unanswerable"]
refused_anyway = [r for r in answerable if abstained(r["answer"])]
had_gold = [r for r in refused_anyway if contains_gold(cases[r["id"]], r["answer"])]

print(f"{len(answerable)} answerable cases")
print(f"  replies the refusal detector fired on      : {len(refused_anyway)}")
print(f"  ...that contain the right answer all the same: {len(had_gold)}"
      "   (scored wrong)")
for r in had_gold[:3]:
    sentence = next((s for s in re.split(r"(?<=[.!?])\s+", r["answer"])
                     if any(p in plain(s) for p in triggers(r["answer"]))), "")
    print(f"    {r['id']}: fired on {triggers(r['answer'])[:2]}")
    print(f"      \"{' '.join(sentence.split())[:84]}\"")

unanswerable = [r for r in rows if cases[r["id"]]["kind"] == "unanswerable"]
passed = [r for r in unanswerable if abstained(r["answer"])]
with_figures = [r for r in passed if re.search(r"\d[\d,]{3,}", r["answer"])]
print(f"\n{len(unanswerable)} cases that must be refused")
print(f"  scored as refusals                          : {len(passed)}")
print(f"  ...that also state a figure of four digits or more: {len(with_figures)}"
      "   (read these)")
for r in with_figures[:2]:
    print(f"    {r['id']}: {' '.join(r['answer'].split())[:84]}")

print("\nA refusal found by phrase-matching prose is a guess about the prose. The fix is")
print("upstream: make abstaining a field the system fills in, not a sentence it writes.")
