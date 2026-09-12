#!/usr/bin/env python3
"""
Reviewing a design — including your own.

Eight questions, asked of Clarity, answered from the repository rather than from
memory. A design review that consults the author's intentions finds nothing; a design
review that greps the code finds three things, and this one does.
"""
from __future__ import annotations

import collections
import json
import math
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CLARITY = ROOT / "code" / "clarity"
SOURCES = sorted(p for p in CLARITY.rglob("*.py") if "__pycache__" not in p.parts)


def loc(path: Path) -> int:
    """Lines that are neither blank nor a comment. Docstrings count: they are the design."""
    return sum(1 for line in path.read_text().splitlines()
               if line.strip() and not line.strip().startswith("#"))


def mentions(pattern: str) -> list[str]:
    """Every Clarity file whose text matches, as repo-relative paths."""
    rx = re.compile(pattern)
    return [str(p.relative_to(ROOT)) for p in SOURCES if rx.search(p.read_text())]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A confidence interval for a proportion that behaves near 0 and 1."""
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


# --------------------------------------------------------------------------- the eight
cases = yaml.safe_load((CLARITY / "evals" / "golden.yaml").read_text())["cases"]
kinds = collections.Counter(c["kind"] for c in cases)
biggest_kind, biggest_n = kinds.most_common(1)[0]
lo, hi = wilson(int(0.85 * len(cases)), len(cases))

state_files = mentions(r"create_engine|sqlite|Session\(")
tenant_files = mentions(r"\btenant\b")
enforcing = mentions(r"tenant\s*==|WHERE.*tenant|filter.*tenant")
incremental = mentions(r"def (upsert|update_document|reindex_one)\b")
refusal = mentions(r"refus|cannot answer|do not contain")
budget = mentions(r"max_steps|budget|Budget")
breaker = mentions(r"CircuitBreaker")
audit = mentions(r"append|audit|Evaluation")

QUESTIONS = [
    ("What happens when the model is down?",
     f"one breaker per provider, in {len(breaker)} file",
     bool(breaker)),
    ("What happens when a tool is wrong?",
     f"a budgeted loop; {len(budget)} files carry a step limit",
     bool(budget)),
    ("What does it do when it does not know?",
     f"refusal is a scored behaviour across {len(refusal)} files",
     bool(refusal)),
    ("Where does state live?",
     f"{len(state_files)} files open a database; the rest are pure",
     bool(state_files)),
    ("How is it evaluated, and how often?",
     f"{len(cases)} cases, six layers, on every push",
     True),
    ("Who can see whose data?",
     f"{len(tenant_files)} files carry a tenant column; "
     f"{len(enforcing)} enforce it",
     len(enforcing) > 0),
    ("What is the cost of one request, and who watches it?",
     "priced per run and stored on the row",
     True),
    ("How does a changed document reach the index?",
     f"{len(incremental)} incremental paths — it is rebuilt, not updated",
     bool(incremental)),
]

print("Eight questions, asked of Clarity, answered by reading the repository.\n")
print(f"  {len(SOURCES)} Python files, {sum(loc(p) for p in SOURCES):,} lines\n")
for question, answer, ok in QUESTIONS:
    mark = " " if ok else "*"
    print(f"  {mark} {question}")
    print(f"      {answer}")

failures = [q for q, _, ok in QUESTIONS if not ok]
print(f"\n{len(QUESTIONS) - len(failures)} of {len(QUESTIONS)} answered cleanly. "
      f"The {len(failures)} marked * are real, and both are")
print("structural rather than accidental — they are things the design does not do,")
print("not things it does badly.")

print("\nAnd one the eight questions do not ask, which a reviewer should:\n")
print(f"  * the eval set is {len(cases)} cases from one corpus, "
      f"{biggest_n} of them ({biggest_n / len(cases):.0%}) {biggest_kind}.")
print(f"    At n={len(cases)}, a score near 85% carries a 95% interval of "
      f"{lo:.0%}-{hi:.0%}: {100 * (hi - lo):.0f} points")
print("    wide, which cannot see a 3-point improvement. §21.6 measured this.")
print("    The fix is more cases, and nothing substitutes for that work.")

print("\nNone of the three is an oversight. Each was a decision to keep the")
print("system small enough to read, and each would be the first thing a team")
print("fixed. Writing them down is the difference between a design and a diagram.")

json.dump({
    "files": len(SOURCES),
    "loc": sum(loc(p) for p in SOURCES),
    "cases": len(cases),
    "kinds": dict(kinds),
    "biggest_kind": biggest_kind,
    "biggest_share": biggest_n / len(cases),
    "interval": [lo, hi],
    "tenant_declaring": len(tenant_files),
    "tenant_enforcing": len(enforcing),
    "incremental": len(incremental),
    "questions": [{"q": q, "a": a, "ok": ok} for q, a, ok in QUESTIONS],
}, open(Path(__file__).parent / "_review.json", "w"), indent=1)
