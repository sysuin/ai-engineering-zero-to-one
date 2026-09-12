#!/usr/bin/env python3
"""
Run every code listing in the book and capture what it actually printed.

The rule this enforces: **no expected output in this book was typed by a human.**
Every listing is a real file; this runner executes it and writes its real stdout and
stderr next to it as a `.out` file, which the manuscript then includes. A listing whose
output was hand-written is a listing that is quietly wrong within two months.

    python3 code/_runner.py              # run everything that changed
    python3 code/_runner.py --all        # ignore the cache, run everything
    python3 code/_runner.py 02           # just chapter 02
    python3 code/_runner.py --check      # fail if any .out is stale (for CI)

Conventions a listing may declare on its first few lines:

    # expect-fail          this listing is supposed to raise; a clean exit is the failure
    # timeout: 300         override the default 120-second limit
    # skip                 do not run (a fragment, or something needing manual setup)

Listings run with the project root as the working directory, so every path in the book
reads the same way: `data/meridian/...`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time

CODE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(CODE)
REPORT = os.path.join(CODE, "_report.json")
DEFAULT_TIMEOUT = 120
CHAPTER_DIR = re.compile(r"^\d{2}$")


def discover(only: str | None) -> list[str]:
    out = []
    for d in sorted(os.listdir(CODE)):
        if not CHAPTER_DIR.match(d):
            continue
        if only and d != only:
            continue
        for f in sorted(os.listdir(os.path.join(CODE, d))):
            if f.endswith(".py") and not f.startswith("_"):
                out.append(os.path.join(d, f))
    return out


def directives(path: str) -> dict:
    d = {"expect_fail": False, "timeout": DEFAULT_TIMEOUT, "skip": False}
    with open(os.path.join(CODE, path)) as f:
        for line in f.read().split("\n")[:12]:
            s = line.strip()
            if s == "# expect-fail":
                d["expect_fail"] = True
            elif s == "# skip":
                d["skip"] = True
            elif s.startswith("# timeout:"):
                d["timeout"] = int(s.split(":")[1])
    return d


def digest(path: str) -> str:
    with open(os.path.join(CODE, path), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()[:16]


def run_one(path: str, spec: dict) -> dict:
    started = time.time()
    proc = subprocess.run(
        [sys.executable, os.path.join("code", path)],
        cwd=ROOT, capture_output=True, text=True, timeout=spec["timeout"],
        env={**os.environ, "PYTHONUNBUFFERED": "1", "BOOK_RUN": "1",
             "COLUMNS": "88", "PYTHONHASHSEED": "0",
             # sitecustomize.py counts model calls; only the build ever sees it.
             "PYTHONPATH": CODE + os.pathsep + os.environ.get("PYTHONPATH", ""),
             "BOOK_LISTING": path},
    )
    elapsed = time.time() - started
    body = proc.stdout
    if proc.stderr.strip():
        body += ("\n" if body and not body.endswith("\n") else "") + proc.stderr

    # Tracebacks carry absolute paths. The book must not contain the author's home
    # directory, and a reader's output should match the book's on their own machine.
    body = body.replace(ROOT + os.sep, "").replace(ROOT, ".")

    failed = proc.returncode != 0
    ok = failed if spec["expect_fail"] else not failed

    with open(os.path.join(CODE, path[:-3] + ".out"), "w") as f:
        f.write(body.rstrip("\n") + "\n")

    return {"ok": ok, "returncode": proc.returncode, "seconds": round(elapsed, 2),
            "bytes": len(body), "expect_fail": spec["expect_fail"]}


def _report_spend(usage_file: str) -> None:
    """Summarise what the model calls cost, in tokens always and dollars if known."""
    if not os.path.exists(usage_file):
        return
    sys.path.insert(0, CODE)
    try:
        from clarity.config import SPEND_CEILING_USD, rate
    except Exception:                                        # noqa: BLE001
        return

    totals: dict[str, list[int]] = {}
    with open(usage_file) as f:
        for line in f:
            r = json.loads(line)
            t = totals.setdefault(r["model"], [0, 0, 0])
            t[0] += r["prompt_tokens"]
            t[1] += r["completion_tokens"]
            t[2] += 1

    if not totals:
        return
    print("\n  model usage")
    dollars, priced = 0.0, True
    for model, (pin, pout, calls) in sorted(totals.items()):
        rates = rate(model)
        if rates:
            cost = (pin * rates[0] + pout * rates[1]) / 1_000_000
            dollars += cost
            money = f"  ${cost:.4f}"
        else:
            priced = False
            money = "  (no rate set)"
        print(f"    {model:22} {calls:4} calls  "
              f"{pin:>7,} in  {pout:>7,} out{money}")

    if priced:
        print(f"    {'total':22} {'':4}         {'':>7}     {'':>7}      ${dollars:.4f}")
        if dollars > SPEND_CEILING_USD:
            print(f"    WARNING: over the ${SPEND_CEILING_USD:.2f} ceiling in .env")
    else:
        print("    Set RATE_<MODEL>_INPUT / _OUTPUT in .env to see dollars.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chapter", nargs="?", help="two-digit chapter, e.g. 02")
    ap.add_argument("--all", action="store_true", help="ignore the cache")
    ap.add_argument("--check", action="store_true",
                    help="fail if anything is stale; run nothing")
    args = ap.parse_args()

    cache = {}
    if os.path.exists(REPORT):
        cache = json.load(open(REPORT)).get("listings", {})

    listings = discover(args.chapter)
    if not listings:
        print("no listings found")
        return 0

    usage_file = os.path.join(CODE, "_usage.jsonl")
    os.environ["BOOK_USAGE_FILE"] = usage_file
    if args.all and os.path.exists(usage_file):
        os.remove(usage_file)

    stale, ran, skipped, failures = [], 0, 0, []
    total_time = 0.0

    for path in listings:
        spec = directives(path)
        if spec["skip"]:
            skipped += 1
            continue

        h = digest(path)
        outfile = os.path.join(CODE, path[:-3] + ".out")
        fresh = (cache.get(path, {}).get("hash") == h and os.path.exists(outfile))

        if args.check:
            if not fresh:
                stale.append(path)
            continue
        if fresh and not args.all:
            continue

        result = run_one(path, spec)
        cache[path] = {"hash": h, **result}
        ran += 1
        total_time += result["seconds"]
        mark = "ok  " if result["ok"] else "FAIL"
        note = "  (expected to fail)" if spec["expect_fail"] else ""
        print(f"  {mark}  {path:34} {result['seconds']:6.2f}s  "
              f"{result['bytes']:>6} bytes{note}")
        if not result["ok"]:
            failures.append(path)

    if args.check:
        for p in stale:
            print(f"  STALE  {p}")
        print(f"\n{len(stale)} stale of {len(listings)}")
        return 1 if stale else 0

    with open(REPORT, "w") as f:
        json.dump({"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
                   "listings": cache}, f, indent=2, sort_keys=True)

    print(f"\n  ran {ran}, cached {len(listings) - ran - skipped}, "
          f"skipped {skipped}, {total_time:.1f}s total")
    _report_spend(usage_file)
    if failures:
        print(f"  FAILURES: {', '.join(failures)}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
