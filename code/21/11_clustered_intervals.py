# Cases that share a template are not independent evidence. Resample templates instead of cases,
# and see how much wider the honest interval is: on the real scorecard, and on a set whose
# failures cluster the way templated failures usually do.

import json
import re
from collections import defaultdict

import numpy as np

rows = json.load(open("code/21/_scorecard.json"))
rng = np.random.default_rng(21)
DRAWS = 5_000


def template(question: str) -> str:
    q = re.sub(r"\b20\d\d\b", "<year>", question)
    q = re.sub(r"\bQ[1-4]\b", "<quarter>", q)
    q = re.sub(r"MSC-\d{4}-\d{3}", "<contract>", q)
    q = re.sub(r"\b(Northeast|Southeast|Midwest|West|Southwest)\b", "<region>", q)
    return re.sub(r"\d[\d,.]*", "<n>", q)


def intervals(passed: np.ndarray, groups: list[str]) -> tuple[tuple, tuple, float]:
    n = len(passed)
    by_case = [passed[rng.integers(0, n, n)].mean() for _ in range(DRAWS)]
    clusters = defaultdict(list)
    for ok, g in zip(passed, groups):
        clusters[g].append(ok)
    members = list(clusters.values())
    by_template = []
    for _ in range(DRAWS):
        draw = rng.integers(0, len(members), len(members))
        picked = [members[i] for i in draw]
        by_template.append(np.mean([ok for m in picked for ok in m]))
    ratio = np.var(by_template) / np.var(by_case)    # the design effect
    return (tuple(np.percentile(by_case, [2.5, 97.5])),
            tuple(np.percentile(by_template, [2.5, 97.5])), ratio)


def show(label: str, passed: np.ndarray, groups: list[str]) -> None:
    (a, b), (c, d), deff = intervals(passed, groups)
    print(f"  {label:32}{passed.mean():>6.0%}{f'{a:.0%}-{b:.0%}':>11}{f'{c:.0%}-{d:.0%}':>13}"
          f"{deff:>8.1f}{len(passed) / deff:>8.0f}")


groups = [template(r["question"]) for r in rows]
passed = np.array([r["correct"] for r in rows])
print(f"{len(rows)} cases in {len(set(groups))} templates; 95% intervals from {DRAWS:,} resamples\n")
print(f"  {'':32}{'score':>6}{'by case':>11}{'by template':>13}{'effect':>8}{'eff. n':>8}")
show("the scorecard as measured", passed, groups)

# A set as good as that one, except that its failures are placed the way a broken template
# places them — every case of the two largest head templates failing together — and the same
# number of failures scattered at random.
head = [g for g, r in zip(groups, rows) if r["tier"] == "head"]
broken = sorted(set(head), key=head.count, reverse=True)[:2]
clustered = np.array([g not in broken for g in groups])
scattered = np.ones(len(rows), dtype=bool)
scattered[rng.choice(len(rows), (~clustered).sum(), replace=False)] = False
show(f"{(~clustered).sum()} failures in two templates", clustered, groups)
show(f"{(~scattered).sum()} failures scattered at random", scattered, groups)

print("\neffect: how many times larger the variance is when templates are resampled;")
print("eff. n: the independent cases the set is worth, the case count divided by it")
