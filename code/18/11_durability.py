# timeout: 300
# LangGraph's three durability modes, on a file-backed SQLite store: what each costs per step,
# and what each leaves behind when the process dies halfway.

import sqlite3
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(tempfile.mkdtemp())
CHILD = "code/18/_durability_child.py"
MODES = {"exit": "written once, when the run ends",
         "async": "written while the next step starts (the default)",
         "sync": "written before the next step starts"}


def survived(store: Path) -> int:
    if not store.exists():
        return 0
    with sqlite3.connect(store) as db:
        try:
            return db.execute("SELECT MAX(json_extract(metadata, '$.step')) FROM checkpoints").fetchone()[0] or 0
        except sqlite3.OperationalError:
            return 0


print(f"  {'durability':10}{'ms per step':>12}{'saved at the crash':>20}{'if steps take 20 ms':>21}")
for mode, when in MODES.items():
    timings = []
    for attempt in range(3):
        store = HERE / f"time-{mode}-{attempt}.db"
        out = subprocess.run([sys.executable, CHILD, str(store), mode, "run"],
                             capture_output=True, text=True, check=True).stdout
        timings.append(float(out.strip()))
    saved = []
    for pause in ("0", "20"):
        crashed = HERE / f"crash-{mode}-{pause}.db"
        subprocess.run([sys.executable, CHILD, str(crashed), mode, "crash", pause], capture_output=True)
        saved.append(survived(crashed))
    print(f"  {mode:10}{statistics.median(timings):>12.2f}{saved[0]:>20}{saved[1]:>21}")

print("\na 200-step graph killed at step 100; 'saved' is the last step a new process could resume from")
for mode, when in MODES.items():
    print(f"  {mode:6} state {when}")
