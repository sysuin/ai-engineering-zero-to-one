# What is actually in Meridian, read from the dataset rather than from memory.

import json
import re
import sqlite3
from pathlib import Path

ROOT = Path("data/meridian")
manifest = json.loads((ROOT / "manifest.json").read_text())

print(f"{manifest['name']}, seed {manifest['seed']}.")
print(f"synthetic: {manifest['synthetic']}   "
      f"contains personal data: {manifest['contains_personal_data']}\n")

print("Counts\n")
for name, n in manifest["counts"].items():
    print(f"  {name.replace('_', ' '):<22}{n:>9,}")

# ------------------------------------------------------------------ the warehouse
db = sqlite3.connect(ROOT / "warehouse" / "meridian.db")
tables = [r[0] for r in db.execute(
    "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
    "AND name NOT LIKE 'sqlite_%' ORDER BY name")]
print(f"\nThe warehouse: {len(tables)} tables and views\n")
for name in tables:
    n = db.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
    cols = [c[1] for c in db.execute(f"PRAGMA table_info({name})")]
    print(f"  {name:<16}{n:>9,} rows   {', '.join(cols[:5])}"
          f"{'…' if len(cols) > 5 else ''}")

# ------------------------------------------------------------------ the documents
docs = ROOT / "documents"
print("\nThe documents. The `-pdf` folders are renders of the same documents, which is")
print("why §13.3 can compare a parser against the text it was generated from.\n")
for folder in sorted(p for p in docs.iterdir() if p.is_dir()):
    files = [p for p in folder.rglob("*")
             if p.is_file() and p.name != "README.md"]
    size = sum(p.stat().st_size for p in files)
    label = "file " if len(files) == 1 else "files"
    print(f"  {folder.name:<22}{len(files):>4} {label}{size / 1e6:>9.1f} MB")

# ------------------------------------------------------------------ the invariant
print("\nThe invariant that makes this dataset worth using:\n")
print("  every figure in a quarterly review is computed from the warehouse,")
print("  so a question can need both and neither alone will answer it.\n")

quarters = manifest["quarterly_totals"]
print(f"  {'quarter':<10}{'manifest revenue':>18}{'warehouse revenue':>19}  agree?")
mismatches = 0
for quarter, totals in list(quarters.items())[:4]:
    year, q = quarter.split()
    stated = totals["revenue"] if isinstance(totals, dict) else totals
    actual = db.execute(
        "SELECT ROUND(SUM(revenue), 2) FROM v_sales WHERE year = ? AND quarter = ?",
        (int(year), int(q.lstrip("Q")))).fetchone()[0]
    agree = abs(stated - actual) < 1.0
    mismatches += not agree
    print(f"  {quarter:<10}{stated:>18,.2f}{actual:>19,.2f}  "
          f"{'yes' if agree else 'NO'}")

checked = min(4, len(quarters))
print(f"\n  {checked - mismatches} of {checked} agree to within a dollar. "
      f"`verify.py` checks all {len(quarters)} and")
print("  fails the build if any of them drift — which is what makes it safe for a")
print("  chapter to quote a figure and for an eval case to be scored against one.")

# ------------------------------------------------------------------ what to be careful of
print("\nThree things in here that are hostile on purpose:\n")


def sources(folder: Path) -> list[str]:
    """Source documents only. The `-pdf` renders and the .pdf siblings are the same
    documents in another format, and counting them twice overstates the corpus."""
    return sorted(p.name for p in folder.iterdir()
                  if p.is_file() and p.suffix in {".md", ".txt", ".jsonl", ".csv"}
                  and p.name != "README.md")


awkward, poisoned = sources(docs / "awkward"), sources(docs / "poisoned")
print(f"  {len(awkward)} awkward documents")
for name in awkward:
    print(f"      {name}")
print(f"\n  {len(poisoned)} poisoned documents, each carrying a real injection payload")
for name in poisoned:
    print(f"      {name}")

# Near-identical by construction, and measured rather than asserted. The header differs
# in every contract — reference, supplier, dates — so the comparison is on the body with
# every number masked, which is exactly what an embedding sees.
contracts = sorted((docs / "contracts").glob("*.md"))
clusters: dict[str, list[str]] = {}
for path in contracts:
    body = path.read_text().split("## 1. Scope", 1)[-1]
    key = re.sub(r"\d+", "#", " ".join(body.split()))
    clusters.setdefault(key, []).append(path.name)
largest = max(clusters.values(), key=len)
print(f"\n  {len(largest)} of {len(contracts)} contracts have identical bodies once "
      "numbers are masked.")
print("  Eight of those are by design and differ only in their numbers and header;")
print("  the extra one is a coincidence of the generator, which is a fair model of")
print("  a real corpus. §14.10 finds this cluster, §14.7's filter separates it, and the")
print("  reason Chapter 13's exercises ask you to go looking for your own.")
