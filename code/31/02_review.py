#!/usr/bin/env python3
"""
Reviewing a design — including your own.

Eight questions, asked of Clarity v1.0, answered from the repository rather than from
memory. The first version of this script searched for words — "tenant", "breaker" — and
reported that a breaker existed and that no tenant filter did. Both were the wrong
question: a breaker existed and the assembled system never called it, and a tenant filter
existed with no tenant column for it to filter. So this version asks what v1.0 *composes*,
not what the repository contains.
"""
from __future__ import annotations

import ast
import collections
import json
import math
import re
import sqlite3
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CLARITY = ROOT / "code" / "clarity"
ASSEMBLY = CLARITY / "v1_0" / "clarity.py"
SOURCES = sorted(p for p in CLARITY.rglob("*.py") if "__pycache__" not in p.parts)


def loc(path: Path) -> int:
    """Lines that are neither blank nor a comment. Docstrings count: they are the design."""
    return sum(1 for line in path.read_text().splitlines()
               if line.strip() and not line.strip().startswith("#"))


def imported_names(path: Path) -> set[str]:
    """Every name a module imports: what it is built from, as opposed to what exists."""
    return {alias.name for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.ImportFrom) for alias in node.names}


def calls_to(path: Path, name: str) -> list[ast.Call]:
    return [node for node in ast.walk(ast.parse(path.read_text()))
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == name]


def defined_in(name: str) -> list[str]:
    rx = re.compile(rf"^class {name}\b", re.M)
    return [str(p.relative_to(CLARITY)) for p in SOURCES if rx.search(p.read_text())]


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """A confidence interval for a proportion that behaves near 0 and 1."""
    p = successes / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return centre - half, centre + half


def template(question: str) -> str:
    """The same collapse as code/21/08: a question with its particulars removed."""
    q = re.sub(r"\b20\d\d\b", "<year>", question)
    q = re.sub(r"\bQ[1-4]\b", "<quarter>", q)
    q = re.sub(r"MSC-\d{4}-\d{3}", "<contract>", q)
    q = re.sub(r"\b(Northeast|Southeast|Midwest|West|Southwest)\b", "<region>", q)
    return re.sub(r"\d[\d,.]*", "<n>", q)


# --------------------------------------------------------------------------- evidence
uses = imported_names(ASSEMBLY)
source = ASSEMBLY.read_text()

from openai import OpenAI                                           # noqa: E402
sdk = OpenAI(api_key="not-a-key")                                   # no request is made
sdk_retries, sdk_read_timeout = sdk.max_retries, sdk.timeout.read

cases = yaml.safe_load((CLARITY / "evals" / "golden.yaml").read_text())["cases"]
kinds = collections.Counter(c["kind"] for c in cases)
biggest_kind, biggest_n = kinds.most_common(1)[0]
templates = len({template(c["question"]) for c in cases})
lo, hi = wilson(int(0.85 * len(cases)), len(cases))

workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "eval.yml").read_text())
samples = [step["env"]["EVAL_SAMPLE"] for job in workflow["jobs"].values()
           for step in job["steps"] if "EVAL_SAMPLE" in step.get("env", {})]
per_pr, full = (int(s) for s in samples)
sampled = f"{per_pr} cases on every pull request, all {len(cases)} weekly"

warehouse_db = ROOT / "data" / "meridian" / "warehouse" / "meridian.db"
schema = sqlite3.connect(warehouse_db).execute(
    "SELECT sql FROM sqlite_master WHERE type IN ('table', 'view')")
tenant_tables = sum("tenant" in (sql or "") for (sql,) in schema)
warehouse_scoped = any(k.arg == "tenant" for c in calls_to(ASSEMBLY, "Warehouse")
                       for k in c.keywords)
retriever_scoped = "tenant" in (CLARITY / "v0_6" / "retrieve.py").read_text()
cache_scoped = "tenant=self.tenant" in source
incremental = [p for p in SOURCES
               if re.search(r"def (upsert|sync|reindex_one|update_document)\b",
                            p.read_text())]

QUESTIONS = [
    ("What happens when the model is down?",
     f"breaker and fallback exist ({defined_in('Gateway')[0]}); v1.0 composes "
     f"{'them' if 'Gateway' in uses else 'neither'}, so it has the SDK's "
     f"{sdk_retries} retries, a {sdk_read_timeout:.0f}s read timeout on each, "
     "and then an error",
     "Gateway" in uses or "CircuitBreaker" in uses),
    ("What happens when a tool is wrong?",
     "a budget of steps, seconds and tokens; a failed or hung tool becomes a "
     "result the model reads",
     "Budget" in uses),
    ("What does it do when it does not know?",
     f"refuses; scored on {kinds['unanswerable']} unanswerable cases by one "
     "detector, shared with the suite",
     "abstained" in uses),
    ("Where does state live, and what if it is lost?",
     "index and warehouse on disk, read-only; the cache in one process's memory, "
     "and lost with it",
     True),
    ("How is it evaluated, and how often?", sampled, per_pr > 0 and full == 0),
    ("Who can see whose data?",
     f"the cache is keyed by tenant{'' if cache_scoped else ' (not)'}; the warehouse "
     f"has {tenant_tables} tenant columns and the index "
     f"{'has one' if retriever_scoped else 'none'}",
     warehouse_scoped and retriever_scoped),
    ("What does one request cost, and who watches it?",
     "every model call priced on its trace span; the alert on the total is "
     "Chapter 23's, outside v1.0",
     "TracedClient" in uses),
    ("How does a changed document reach the index?",
     f"{len(incremental)} incremental paths: the index is rebuilt whole",
     bool(incremental)),
]

print("Eight questions, asked of Clarity v1.0, answered by reading what it composes.\n")
print(f"  {len(SOURCES)} Python files in the repository, "
      f"{sum(map(loc, SOURCES)):,} lines; v1.0 imports {len(uses)} names\n")
for question, answer, ok in QUESTIONS:
    print(f"  {' ' if ok else '*'} {question}")
    words, line = answer.split(), "     "
    for word in words:
        if len(line) + len(word) > 74:
            print(line)
            line = "     "
        line += " " + word
    print(line)

failures = [q for q, _, ok in QUESTIONS if not ok]
print(f"\n{len(QUESTIONS) - len(failures)} of {len(QUESTIONS)} answered cleanly; "
      f"the {len(failures)} marked * are not.")

# What was built for the system and never reached it: every public name a platform
# module defines, minus what v1.0 imports and what those pieces use in turn.
PLATFORM = sorted((CLARITY / "platform").glob("*.py"))
reached = set().union(*(imported_names(p) for p in (CLARITY / "v1_0").glob("*.py")))
while True:
    before = len(reached)
    for module in PLATFORM:
        tree = ast.parse(module.read_text())
        for node in tree.body:
            if getattr(node, "name", None) in reached:
                reached |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
                reached |= imported_names(module)
    if len(reached) == before:
        break
print("\nBuilt in platform/, and never reached from v1.0:\n")
unused = {}
for module in PLATFORM:
    names = [node.name for node in ast.parse(module.read_text()).body
             if isinstance(node, (ast.ClassDef, ast.FunctionDef))
             and not node.name.startswith("_") and node.name not in reached]
    if names:
        unused[module.stem] = names
        print(f"  {module.stem:<12} {', '.join(names)}")

print("\nAnd one the eight questions do not ask, which a reviewer should:\n")
print(f"  * the eval set is {len(cases)} cases from one corpus, "
      f"{biggest_n} of them ({biggest_n / len(cases):.0%}) {biggest_kind},")
print(f"    and {templates} distinct questions once years and quarters are removed.")
print(f"    At n={len(cases)} a score near 85% has a 95% interval of "
      f"{lo:.0%}-{hi:.0%}, {100 * (hi - lo):.0f} points")
print("    wide — and that assumes every case is independent, which templates are not.")

json.dump({
    "files": len(SOURCES),
    "loc": sum(map(loc, SOURCES)),
    "cases": len(cases),
    "kinds": dict(kinds),
    "templates": templates,
    "biggest_kind": biggest_kind,
    "biggest_share": biggest_n / len(cases),
    "interval": [lo, hi],
    "sdk_retries": sdk_retries,
    "sdk_read_timeout": sdk_read_timeout,
    "tenant_columns_in_warehouse": tenant_tables,
    "incremental": len(incremental),
    "unused": unused,
    "questions": [{"q": q, "a": a, "ok": ok} for q, a, ok in QUESTIONS],
}, open(Path(__file__).parent / "_review.json", "w"), indent=1)
