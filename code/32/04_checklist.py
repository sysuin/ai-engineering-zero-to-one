# Clarity against its own pre-launch checklist, checked by reading the repository.
#
# A checklist ticked from memory is ticked generously. Every line below that a script can
# decide is decided by one; the lines a script cannot decide say so, because pretending
# otherwise is how "someone who did not build it has used it" gets ticked by the builder.

import ast
import json
import re
import subprocess
from pathlib import Path

from openai import OpenAI

ROOT = Path(".")
CLARITY = ROOT / "code" / "clarity"
V1 = CLARITY / "v1_0"


def text(path: Path) -> str:
    return path.read_text()


def imports(path: Path) -> set[str]:
    return {alias.name for node in ast.walk(ast.parse(text(path)))
            if isinstance(node, ast.ImportFrom) for alias in node.names}


composed = set().union(*(imports(p) for p in V1.glob("*.py")))
service, clarity = text(V1 / "service.py"), text(V1 / "clarity.py")
card = json.load(open("code/32/_card.json"))
read_timeout = OpenAI(api_key="not-a-key").timeout.read           # no request is made

# `git grep` over tracked files only: the count of anything shaped like an API key.
# Nothing matched is printed, only how many — a check that echoes secrets is a leak.
secret_like = subprocess.run(
    ["git", "grep", "-cE", r"sk-[A-Za-z0-9_-]{20,}"],
    capture_output=True, text=True).stdout.strip().splitlines()
env_ignored = subprocess.run(["git", "check-ignore", "-q", ".env"]).returncode == 0

PERSON = None          # a line only a person can check
CHECKS = {
    "it works": [
        ("the eval suite is green against a recorded baseline",
         "baseline" in text(ROOT / "tests" / "test_evals.py"),
         ("a baseline is passed to the suite",
          "floors only: suite.py can compare to a baseline; none is recorded")),
        ("the red-team suite runs and its failures are known",
         "blocked" in card,
         (f"{card['blocked']:.0%} of attacks blocked on the last card", "no card yet")),
        ("someone who did not build it has used it", PERSON, ("", "")),
    ],
    "it fails safely": [
        ("every dependency has a timeout",
         read_timeout <= 60,
         (f"read timeout {read_timeout:.0f}s",
          f"the model client's read timeout is the SDK default, {read_timeout:.0f}s")),
        ("there is a breaker per provider, not one shared",
         "Gateway" in composed or "CircuitBreaker" in composed,
         ("composed by v1.0", "built in platform/, not composed by v1.0")),
        ("a refusal is a designed answer, not an exception",
         "abstained" in composed, ("one detector, shared with the suite", "")),
        ("the load-shedding path returns 503, not a hang",
         "status_code=503" in service, ("the bulkhead's Open becomes a 503", "")),
    ],
    "you can see it": [
        ("every answer carries a trace id you can search",
         "trace_id" in service and "start(" in service, ("on every response", "")),
        ("cost per request is recorded on the row",
         "open_store" in service or "open_store" in clarity,
         ("a run row is written", "priced on spans; v1.0 writes no row")),
        ("an alert fires on quality, not only on errors",
         bool(re.search(r"alert", clarity + service, re.I)),
         ("", "no alerting in the code")),
    ],
    "it is contained": [
        ("no secret is in any file git can see",
         not secret_like and env_ignored,
         2 * (f"{len(secret_like)} tracked files match a key pattern; "
              f".env {'is' if env_ignored else 'is NOT'} ignored",)),
        ("untrusted text is fenced before it reaches a prompt",
         "fence" in composed, ("", "guard.fence exists; v1.0 does not call it")),
        ("the tenant boundary has a test that fails without it",
         "tenant" in text(ROOT / "tests" / "test_evals.py"),
         ("", "Chapter 26's test runs on its own listing, not on Clarity")),
        ("there is a spend ceiling and it has been tested",
         bool(re.search(r"SPEND_CEILING|spend_ceiling", clarity + service)),
         ("", "the book's runner has one; Clarity does not")),
    ],
    "someone can run it": [
        ("one command starts it from a clean checkout", PERSON, ("", "")),
        ("the README's first five commands work verbatim", PERSON, ("", "")),
        ("the known weaknesses are written down",
         "Known weaknesses" in text(ROOT / "CONTRIBUTING.md"),
         ("CONTRIBUTING.md, from Chapter 31", "")),
    ],
}

tally = {"yes": 0, "no": 0, "person": 0}
for group, items in CHECKS.items():
    print(f"\n  {group}")
    for item, ok, notes in items:
        note = notes[0] if ok else notes[1]
        mark = {True: "yes", False: " no", None: "  ?"}[ok]
        tally["yes" if ok else "person" if ok is None else "no"] += 1
        print(f"    {mark}  {item}")
        if note:
            print(f"           {note}")

print(f"\n{tally['yes']} yes, {tally['no']} no, {tally['person']} for a person to check.")
json.dump(tally, open("code/32/_checklist.json", "w"), indent=1)
