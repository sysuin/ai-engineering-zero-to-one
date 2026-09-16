# timeout: 1800
# Depends on clarity/evals/runner.py and clarity/evals/judge.py.
# The judge was measured on answers a string rule can already score. Here are answers it
# cannot: the right figure written the way people write figures.

import json
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "code")
sys.path.insert(0, "code/22")
from _figures import build                                      # noqa: E402
from clarity.evals.judge import REFERENCE, Judge, kappa         # noqa: E402
from clarity.evals.runner import correct                        # noqa: E402

rows = build()
cases = {r["id"] for r in rows}
judge = Judge(system=REFERENCE)
with ThreadPoolExecutor(max_workers=12) as pool:
    verdicts = list(pool.map(lambda r: judge.score(r["question"], r["answer"],
                                                   r["expected"]).correct, rows))

labels = [r["label"] for r in rows]
string_rule = [int(correct(r["case"], r["answer"])) for r in rows]
judged = [int(bool(v)) for v in verdicts]

print(f"{len(cases)} figures of six digits or more, each written four ways:")
print("the right figure and a wrong one, in millions and in rounded thousands\n")
print(f"  {'':22} {'agreement':>9} {'kappa':>6} {'right ones passed':>18} {'wrong ones passed':>18}")
for name, marks in (("string rule", string_rule), ("reference judge", judged)):
    agree, k = kappa(labels, marks)
    right = sum(m for m, l in zip(marks, labels) if l)
    wrong = sum(m for m, l in zip(marks, labels) if not l)
    print(f"  {name:22} {agree:>8.0%} {k:>6.2f} {right:>12} of {sum(labels):<4}"
          f"{wrong:>12} of {len(labels) - sum(labels)}")

for form in ("millions", "thousands"):
    subset = [(j, l) for j, l, r in zip(judged, labels, rows) if r["form"] == form]
    print(f"\n  judge on '{form}': {sum(j == l for j, l in subset)} of {len(subset)} "
          f"agree with the label")
disagreements = [(r, j) for r, j in zip(rows, judged) if j != r["label"]]
for r, j in disagreements[:3]:
    print(f"    expected {r['expected']}, answer {r['answer']!r} -> judge said "
          f"{'CORRECT' if j else 'WRONG'}")

json.dump({"labels": labels, "string": string_rule, "judge": judged,
           "forms": [r["form"] for r in rows]},
          open("code/22/_earns.json", "w"), indent=1)
