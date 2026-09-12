# skip
"""
A balanced set of answers with labels nobody has to argue about.

Chapter 21's run produced 110 correct answers out of 120, which is a fine result and a
useless calibration set: a judge that says CORRECT to everything scores 92%, and kappa
on ten negatives is a rumour.

So we build the negatives. Each one takes a real Clarity answer and breaks exactly one
thing — the figure, or the named entity — leaving the prose, the length and the
confidence untouched. The label is then certain by construction rather than by
somebody's opinion, which is the only way to measure a judge without first solving the
problem the judge is for.

Four variants per case, and each exists to isolate one thing:

    plain      the answer Clarity gave
    wrong      one fact changed; still fluent, still confident
    padded     the same correct answer with 60 words of true, irrelevant context
    terse      the same correct answer reduced to the bare fact
"""
from __future__ import annotations

import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from clarity.evals.runner import correct, load                 # noqa: E402

random.seed(22)

PADDING = (
    " For context, Meridian Supply Co. is a distributor operating across five "
    "regions, and its quarterly business reviews are prepared by Commercial "
    "Analytics and classified as internal management reporting. The reviews are "
    "not audited. Figures are stated gross of discount and exclude tax and freight, "
    "and the standing gross margin target across the business is 32.0%.")

ENTITIES = ["Halloway Group", "Voss Industrial", "Pemberton Mills", "Midwest",
            "Southwest", "Northeast", "Southeast", "West", "Safety", "Sanitation",
            "Packaging", "Cleaning", "Facilities"]


def corrupt(answer: str, case: dict) -> str | None:
    """
    Change the fact the question asks about, and nothing else.

    The first version of this function changed the longest number in the reply. For
    "what was gross margin in 2023 Q1?" that number is the *year*, so it produced
    "gross margin in 2,366 Q1 was 31.0%" — an answer that still states the correct
    fact, labelled as wrong. Fifty-one of those went into the calibration set and made
    the judge look far worse than it was.

    So this version targets the expected answer itself, wherever it appears in the
    reply, and `verified()` below refuses any corruption that leaves it behind.
    """
    expected = str(case.get("expected") or case.get("answer") or "").strip()
    if not expected:
        return None

    forms = {expected, expected.replace(",", ""), expected.rstrip("%")}
    for form in sorted(forms, key=len, reverse=True):
        if not form or form not in answer:
            continue
        try:
            value = float(form.replace(",", "").rstrip("%"))
        except ValueError:
            break
        moved = value * 1.17 if abs(value) > 100 else value + 7
        replacement = (f"{moved:,.2f}" if "." in form else f"{int(round(moved)):,}")
        if form.endswith("%"):
            replacement += "%"
        return answer.replace(form, replacement, 1)

    # A named answer: swap it for another name from the same corpus.
    if expected in ENTITIES or any(e.lower() == expected.lower() for e in ENTITIES):
        others = [e for e in ENTITIES if e.lower() != expected.lower()]
        return re.sub(re.escape(expected), random.choice(others), answer,
                      count=1, flags=re.I)

    # The number the model actually wrote, which may be more precise than the gold.
    numbers = [n for n in re.findall(r"\d[\d,]*\.\d+", answer)]
    if numbers:
        target = max(numbers, key=len)
        value = float(target.replace(",", ""))
        return answer.replace(target, f"{value * 1.17:,.2f}", 1)
    return None


def terse(answer: str, case: dict) -> str:
    return f"{case['answer']}."


def neighbour(answer: str, case: dict, pool: dict[str, list[str]]) -> str | None:
    """
    The hard negative: a real figure from the corpus, in the wrong place.

    Substituting a made-up number tests whether a judge can compare two strings.
    Substituting the *previous quarter's actual revenue* tests whether it is checking
    the question or recognising a plausible Meridian figure — which is the mistake a
    judge makes when it is grading vibes.
    """
    expected = str(case.get("expected") or case.get("answer") or "").strip()
    kind = "".join(ch for ch in case["id"] if not ch.isdigit()).rstrip("-Q")
    candidates = [v for v in pool.get(kind, []) if v and v != expected]
    if not candidates or expected not in answer:
        return None
    return answer.replace(expected, random.choice(candidates), 1)


def verified(broken: str, case: dict) -> bool:
    """
    Prove the negative is negative.

    A corrupted answer that still contains the expected fact is not a wrong answer,
    it is a mislabelled one — and a judge measured against mislabelled negatives will
    look broken while being right. This is §21.4's span verifier, pointed at labels
    instead of sources.
    """
    return not correct(case, broken)


def build() -> list[dict]:
    cases = {c["id"]: c for c in load()}
    rows = json.load(open("code/21/_scorecard.json"))
    # Every real answer of the same shape, so a swap lands on a genuine Meridian fact.
    pool: dict[str, list[str]] = {}
    for case in cases.values():
        if case.get("answer"):
            key = "".join(ch for ch in case["id"] if not ch.isdigit()).rstrip("-Q")
            pool.setdefault(key, []).append(case["answer"])
    out: list[dict] = []
    for row in rows:
        case = cases[row["id"]]
        # Only cases Clarity actually got right, so "plain" is reliably a positive.
        if not row["correct"] or case["kind"] == "unanswerable":
            continue
        answer = row["answer"].strip()
        broken = corrupt(answer, case)
        if broken is None or broken == answer or not verified(broken, case):
            continue
        out.append(dict(id=row["id"], variant="plain", label=1, kind=case["kind"],
                        question=case["question"], expected=case["answer"],
                        answer=answer))
        out.append(dict(id=row["id"], variant="wrong", label=0, kind=case["kind"],
                        question=case["question"], expected=case["answer"],
                        answer=broken))
        out.append(dict(id=row["id"], variant="padded", label=1, kind=case["kind"],
                        question=case["question"], expected=case["answer"],
                        answer=answer + PADDING))
        out.append(dict(id=row["id"], variant="terse", label=1, kind=case["kind"],
                        question=case["question"], expected=case["answer"],
                        answer=terse(answer, case)))
        swapped = neighbour(answer, case | {"expected": case["answer"]}, pool)
        if swapped and swapped != answer and verified(swapped, case):
            out.append(dict(id=row["id"], variant="neighbour", label=0,
                            kind=case["kind"], question=case["question"],
                            expected=case["answer"], answer=swapped))
    return out


if __name__ == "__main__":
    rows = build()
    Path("code/22/_answerset.json").write_text(json.dumps(rows, indent=1))
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["variant"]] = counts.get(row["variant"], 0) + 1
    print(f"{len(rows)} answers over {len(rows) // 4} cases")
    print(f"  {counts}")
    print(f"  positives {sum(r['label'] for r in rows)}, "
          f"negatives {sum(1 - r['label'] for r in rows)}")
    print("\nOne case, all four variants:\n")
    first = rows[0]["id"]
    for row in [r for r in rows if r["id"] == first]:
        print(f"  {row['variant']:<8} label={row['label']}  "
              f"{' '.join(row['answer'].split())[:78]}")
