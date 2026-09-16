# timeout: 1800
# Depends on clarity/evals/runner.py; re-run when the scorer changes.
# Reads code/22/_answerset.json, built by _answers.py — the real answers, each
# paired with one broken on purpose and verified to be wrong.
# A model grading a model, measured against labels that are certain.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
from clarity.evals.judge import POINTWISE, REFERENCE, Judge, kappa   # noqa: E402

rows = [r for r in json.load(open("code/22/_answerset.json"))
        if r["variant"] in ("plain", "wrong", "neighbour")]
labels = [r["label"] for r in rows]


def run(judge: Judge) -> list[int]:
    def one(row):
        expected = row["expected"] if judge.system is REFERENCE else None
        return judge.score(row["question"], row["answer"], expected)
    with ThreadPoolExecutor(max_workers=8) as pool:
        return [1 if v.correct else 0 for v in pool.map(one, rows)]


negatives = sum(1 for r in rows if not r["label"])
print(f"{len(rows)} answers: {sum(r['label'] for r in rows)} real ones, "
      f"{negatives} broken on purpose.")
print("Every negative is verified to no longer contain the fact it should.\n")
print("  invented   a number that is not in the corpus at all")
print("  neighbour  a real Meridian figure, from the wrong quarter\n")

results = {}
for name, system in (("no reference", POINTWISE), ("with the reference", REFERENCE)):
    votes = run(Judge(system=system))
    agreement, k = kappa(votes, labels)
    passed = {}
    for variant in ("wrong", "neighbour"):
        idx = [i for i, r in enumerate(rows) if r["variant"] == variant]
        passed[variant] = (sum(votes[i] for i in idx), len(idx))
    failed_good = sum(1 for v, r in zip(votes, rows) if not v and r["label"])
    results[name] = {"votes": votes, "agreement": agreement, "kappa": k,
                     "passed": passed, "failed_good": failed_good}
    print(f"  {name}")
    print(f"    agreement with the label   {agreement:.0%}")
    print(f"    Cohen's kappa              {k:.2f}")
    for variant, (through, total) in passed.items():
        print(f"    {variant + ' negatives passed':<27}{through} of {total}")
    print(f"    {'correct answers failed':<27}{failed_good} of "
          f"{sum(r['label'] for r in rows)}")
    print()

json.dump({"labels": labels, "ids": [r["id"] for r in rows],
           "variants": [r["variant"] for r in rows], **results},
          open("code/22/_judge.json", "w"), indent=1)

bare, ref = results["no reference"], results["with the reference"]
bare_through = sum(v[0] for v in bare["passed"].values())
ref_through = sum(v[0] for v in ref["passed"].values())
print(f"Without the reference the judge let {bare_through} of {negatives} wrong "
      f"answers through.")
print(f"With it, {ref_through}.")
print()
print("That is the difference between grading and guessing. A judge with no reference")
print("is being asked whether an answer sounds right about a company it has never")
print("heard of, and a confident sentence with a wrong number in it sounds exactly as")
print("right as the true one.")
print()
print("Do not read the second row as 'judges work'. Read it as: this judge, on this")
print("task, where correctness is a figure that either matches or does not. Every")
print("negative here is checkable by comparing two strings, which is what the judge")
print("did — and if that is your task, §22.13 has a cheaper suggestion than a model.")
