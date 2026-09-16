# A checkpoint is written after a node finishes. A process that dies inside a node,
# after its side effect but before its checkpoint, runs that node again on resume.
# Durable execution is at-least-once; exactly-once is something you build.

import json
import sqlite3
import subprocess
import sys
from pathlib import Path

STORE, LEDGER = Path("data/meridian/refunds.db"), Path("data/meridian/ledger.jsonl")

CHILD = "code/18/_refund_child.py"


def clean():
    for path in (STORE, LEDGER, Path(f"{STORE}-wal"), Path(f"{STORE}-shm")):
        path.unlink(missing_ok=True)


for mode in ("naive", "idempotent"):
    clean()
    run = [sys.executable, CHILD, str(STORE), str(LEDGER), mode]
    first = subprocess.run(run + ["crash"], capture_output=True, text=True)
    with sqlite3.connect(STORE) as con:
        rows = con.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
    second = subprocess.run(run + ["resume"], capture_output=True, text=True)
    entries = [json.loads(line) for line in LEDGER.read_text().splitlines()]
    print(f"{mode}:")
    print(f"  first process exit code {first.returncode}, {rows} checkpoints written")
    print(f"  second process: {second.stdout.strip() or second.stderr.strip()[-80:]}")
    print(f"  refunds in the ledger: {len(entries)}, "
          f"total {sum(e['amount'] for e in entries):,.2f}\n")

clean()
