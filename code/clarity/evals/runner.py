"""
Clarity's eval harness: run the golden set, score it, and say where it hurts.

Two rules hold this together.

**Score by kind, never in aggregate alone.** One number over 120 mixed cases hides
everything: a system that is perfect on the warehouse and hopeless on contracts scores
the same as one that is mediocre at both, and only one of those is fixable this week.

**An abstention is a right answer.** Twenty of these cases have no answer in the
corpus. Counting them as failures because the system said "I don't know" is how eval
sets teach systems to guess.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

GOLDEN = Path(__file__).parent / "golden.yaml"
REFUSALS = ("i don't know", "i do not know", "not in the", "no information",
            "cannot be answered", "can't be answered", "does not appear",
            "not available", "unable to", "no data", "not stated", "not found",
            "nothing in the", "could not find", "couldn't find", "not covered",
            "no mention", "not something", "not present", "insufficient",
            "do not answer that", "does not answer", "no answer", "don't have",
            "do not have", "not disclosed", "not specified",
            # Added in Chapter 27, where a second model phrased every refusal as
            # "the passages do not contain X" and scored 0 of 12. A refusal detector
            # built on string matching meets a new phrasing every time the wording
            # of a prompt changes — §21.5's apostrophe, again.
            "do not contain", "does not contain", "don't contain",
            "not contain", "no mention of", "cannot determine")


def load() -> list[dict]:
    return yaml.safe_load(GOLDEN.read_text())["cases"]


# Models write curly apostrophes and dashes. The first version of this file did not,
# and scored every one of Clarity's twenty refusals as a failure — 0 out of 20, on a
# system that was refusing correctly every time. §21.5 is that afternoon.
PUNCTUATION = str.maketrans({"\u2019": "'", "\u2018": "'", "\u201c": '"',
                             "\u201d": '"', "\u2013": "-", "\u2014": "-",
                             "\u00a0": " "})


def plain(text: str) -> str:
    return (text or "").translate(PUNCTUATION).lower()


def normalise(text: str) -> str:
    return re.sub(r"[\s,$%]+", "", plain(text))


def abstained(answer: str) -> bool:
    return any(phrase in plain(answer) for phrase in REFUSALS)


def correct(case: dict, answer: str) -> bool:
    """
    Did the system get it right?

    For an unanswerable case, right means refusing. For everything else it means the
    answer appears somewhere in the reply — deliberately generous, because this
    chapter is measuring whether the fact was found, not whether the prose was tidy.
    §21.9 measures what that generosity costs.
    """
    if case["kind"] == "unanswerable":
        return abstained(answer)
    if abstained(answer):
        return False
    hay = normalise(answer)
    wanted = [case["answer"]] + list(case.get("accept") or [])
    return any(normalise(w) in hay for w in wanted if w)


@dataclass
class Scorecard:
    rows: list[dict] = field(default_factory=list)

    def add(self, case: dict, answer: str, **extra) -> None:
        self.rows.append({"id": case["id"], "kind": case["kind"],
                          "tier": case["tier"], "question": case["question"],
                          "answer": answer, "correct": correct(case, answer),
                          **extra})

    def by(self, field_name: str) -> dict[str, tuple[int, int]]:
        out: dict[str, list[int]] = {}
        for row in self.rows:
            bucket = out.setdefault(row[field_name], [0, 0])
            bucket[0] += int(row["correct"])
            bucket[1] += 1
        return {k: (v[0], v[1]) for k, v in sorted(out.items())}

    @property
    def score(self) -> float:
        return sum(r["correct"] for r in self.rows) / max(len(self.rows), 1)

    def table(self, field_name: str) -> str:
        lines = []
        for name, (right, total) in self.by(field_name).items():
            bar = "#" * round(right / total * 24)
            lines.append(f"  {name:<14} {right:>3}/{total:<4} {right / total:>5.0%}  "
                         f"{bar}")
        return "\n".join(lines)
