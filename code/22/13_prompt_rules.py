# timeout: 1800
# Two of this chapter's rules for a judge's prompt, tested on its calibration set: does the order
# of reason and verdict change accuracy, and does naming which version wrote an answer change the
# winner? The padded variant is left out; only answers whose labels are certain are used.

import json
from concurrent.futures import ThreadPoolExecutor

import sys
sys.path.insert(0, "code")
from clarity.evals.judge import PAIRWISE, REFERENCE, Judge      # noqa: E402

rows = [r for r in json.load(open("code/22/_answerset.json"))
        if r["variant"] in ("plain", "wrong", "neighbour")]
REASON_FIRST = REFERENCE
VERDICT_FIRST = (REFERENCE.replace('{"reason": "...", "verdict": "CORRECT" or "WRONG"}',
                                   '{"verdict": "CORRECT" or "WRONG", "reason": "..."}')
                 .replace("Write the reason first and the verdict second.",
                          "Write the verdict first and the reason second."))
assert VERDICT_FIRST != REASON_FIRST and VERDICT_FIRST.count("verdict first") == 1

print(f"1. Reason or verdict first: {len(rows)} labelled answers, reference judge\n")
print(f"  {'order':<16}{'agrees':>10}{'no verdict':>12}")
for name, system in (("reason first", REASON_FIRST), ("verdict first", VERDICT_FIRST)):
    judge = Judge(system=system)
    with ThreadPoolExecutor(max_workers=12) as pool:
        verdicts = list(pool.map(lambda r: judge.score(r["question"], r["answer"], r["expected"]), rows))
    agree = sum(v.correct == bool(r["label"]) for v, r in zip(verdicts, rows))
    missing = sum(v.correct is None for v in verdicts)
    print(f"  {name:<16}{agree:>6}/{len(rows)}{missing:>12}")

# 2. Two correct answers to the same question: the plain answer and the terse one.
by_id = {}
for r in json.load(open("code/22/_answerset.json")):
    by_id.setdefault(r["id"], {})[r["variant"]] = r
pairs = [(v["plain"], v["terse"]) for v in by_id.values() if "plain" in v and "terse" in v]
pairwise = Judge(system=PAIRWISE)


def winner(question: str, first: str, second: str, labels: tuple[str, str] | None) -> str:
    if labels:
        first, second = f"({labels[0]}) {first}", f"({labels[1]}) {second}"
    return pairwise.compare(question, first, second)


jobs = []
for plain, terse in pairs:
    for order in ((plain, terse), (terse, plain)):        # both positions, to cancel position bias
        jobs.append((plain["question"], order, plain["answer"]))
CONDITIONS = {"no labels": None,
              "plain answer labelled new": "plain",
              "terse answer labelled new": "terse"}
print(f"\n2. Naming the version: {len(pairs)} pairs of correct answers, each in both positions\n")
print(f"  {'labels':<30}{'plain wins':>11}{'terse wins':>11}{'tie':>6}")
for name, newer in CONDITIONS.items():
    def one(job):
        question, (a, b), plain_text = job
        labels = None
        if newer:
            a_is_new = (a["answer"] == plain_text) == (newer == "plain")
            labels = ("new version", "old version") if a_is_new else ("old version", "new version")
        result = winner(question, a["answer"], b["answer"], labels)
        if result == "TIE":
            return "tie"
        chosen = a if result == "A" else b
        return "plain" if chosen["answer"] == plain_text else "terse"
    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(one, jobs))
    print(f"  {name:<30}{outcomes.count('plain'):>7}/{len(jobs)}{outcomes.count('terse'):>7}/{len(jobs)}"
          f"{outcomes.count('tie'):>6}")
