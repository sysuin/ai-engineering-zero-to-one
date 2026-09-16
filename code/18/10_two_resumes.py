# timeout: 300
# Two workers pick up the same paused run at the same moment. A checkpointer stores state;
# it does not decide who may continue it.

import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(tempfile.mkdtemp())
CHILD = "code/18/_resume_child.py"


def race(lease: bool) -> tuple[int, list[str]]:
    store, ledger = HERE / f"runs-{lease}.db", HERE / f"ledger-{lease}.txt"
    subprocess.run([sys.executable, CHILD, str(store), str(ledger), "pause", "0", "-"], check=True)
    start_at = str(time.time() + 1.0)                     # both wait for the same instant
    workers = [subprocess.Popen([sys.executable, CHILD, str(store), str(ledger), "resume",
                                 start_at, "lease" if lease else "-", name],
                                stdout=subprocess.PIPE, text=True) for name in ("worker-a", "worker-b")]
    said = [w.communicate()[0].strip() for w in workers]
    refunds = ledger.read_text().splitlines() if ledger.exists() else []
    return len(refunds), said


for lease in (False, True):
    n, said = race(lease)
    print(f"{'with a lease on the run' if lease else 'no lease'}:")
    for line in said:
        print(f"  {line}")
    print(f"  refunds written: {n}\n")
