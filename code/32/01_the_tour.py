# The tour: every file in Clarity, what it does, and which chapter argued for it.
#
# Generated rather than written, because a hand-written file index is wrong within a
# month and nobody notices until a new person trusts it.

import ast
import json
from pathlib import Path

ROOT = Path("code/clarity")

# Which chapter each directory came from. This is the only hand-maintained mapping here,
# and it is the one thing a script cannot recover from the source.
ORIGIN = {
    "": 4, "v0_1": 4, "v0_3": 7, "v0_4": 8, "v0_5": 13, "v0_6": 14, "v0_7": 15,
    "v0_8": 16, "v0_9": 17, "v0_10": 18, "v0_11": 19, "v0_12": 20,
    "evals": 21, "redteam": 21, "platform": 23, "v0_17": 25, "prompts": 7,
    "v1_0": 32,
}
LAYER = {"platform": "the platform", "evals": "evaluation",
         "redteam": "evaluation", "v1_0": "the system"}


def summary(path: Path) -> str:
    """The first sentence of the module docstring — which is why every module has one."""
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return ""
    doc = ast.get_docstring(tree) or ""
    first = doc.strip().split("\n")[0]
    # Strip the "Clarity v0.6 — " title prefix, and only that. An earlier version split
    # on the dash unconditionally and turned config.py's summary into the second half
    # of its own sentence.
    if first.startswith("Clarity") and "—" in first:
        first = first.split("—", 1)[1]
    return first.strip()


def body_lines(path: Path) -> int:
    return sum(1 for line in path.read_text().splitlines()
               if line.strip() and not line.strip().startswith("#"))


def folder(path: Path) -> str:
    """The directory that dates a file. `config.py` sits at the root and is Chapter 4's."""
    return path.parent.name if path.parent != ROOT else ""


def clarity_imports(path: Path) -> set[Path]:
    """The Clarity modules one file imports, resolved to files."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not (isinstance(node, ast.ImportFrom) and node.module
                and node.module.startswith("clarity")):
            continue
        base = ROOT.joinpath(*node.module.split(".")[1:])
        for target in [base.with_suffix(".py"), base / "__init__.py",
                       *(base / f"{alias.name}.py" for alias in node.names)]:
            if target.exists():
                found.add(target)
    return found


# What v1.0 actually runs: every file reachable by import from its four entry points.
# The rest of the repository is the book's history — the system as each chapter left it.
reached, pending = set(), list((ROOT / "v1_0").glob("*.py"))
while pending:
    path = pending.pop()
    if path not in reached:
        reached.add(path)
        pending.extend(clarity_imports(path))

candidates = [p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts
              and (p.name != "__init__.py" or body_lines(p) > 20)]
files = sorted(candidates,
               key=lambda p: (ORIGIN.get(folder(p), 99),
                              LAYER.get(folder(p), ""), str(p)))

print("Clarity, file by file, sorted by the chapter that built it.")
print("A * marks the files v1.0 runs; the others are the book's history.")
group = None
rows = []
for path in files:
    directory = folder(path)
    chapter = ORIGIN.get(directory, 0)
    label = LAYER.get(directory, f"chapter {chapter}")
    if label != group:
        print(f"\n  {label}")
        group = label
    text = summary(path)
    runs = path in reached
    # Two files are called service.py and two are called __init__.py. The directory is
    # not decoration here; it is the only thing that tells them apart.
    name = f"{directory}/{path.name}" if directory else path.name
    print(f"  {'*' if runs else ' '} {name:<26}{body_lines(path):>5}  {text[:40]}")
    rows.append({"file": str(path.relative_to(ROOT.parent)), "chapter": chapter,
                 "lines": body_lines(path), "summary": text, "runs": runs})


def split(selected: list[dict]) -> dict:
    total = sum(r["lines"] for r in selected)
    platform = sum(r["lines"] for r in selected if "/platform/" in r["file"])
    evals = sum(r["lines"] for r in selected
                if "/evals/" in r["file"] or "/redteam/" in r["file"])
    return {"files": len(selected), "total": total, "platform": platform,
            "evals": evals, "answering": total - platform - evals}


everything = split(rows)
running = split([r for r in rows if r["runs"]])
print(f"\n{everything['files']} files and {everything['total']:,} lines in the "
      f"repository; v1.0 runs {running['files']} of them,")
print(f"{running['total']:,} lines. Where the running system's lines went:\n")
for name, key in (("answering the question", "answering"),
                  ("the platform around it", "platform"),
                  ("knowing whether it works", "evals")):
    count = running[key]
    bar = "#" * round(40 * count / running["total"])
    print(f"  {name:<26}{count:>6}  {count / running['total']:>4.0%}  {bar}")

share = running["answering"] / running["total"]
unused = sum(r["lines"] for r in rows if "/platform/" in r["file"] and not r["runs"])
print(f"\nThe part that answers the question is {share:.0%} of what runs. The other "
      f"{1 - share:.0%} is")
print("tracing, caching, guarding, shedding load and the suite that says whether the")
print(f"answers are right. {unused:,} more lines of platform — the gateway and the replay")
print("tools — are not in that count, because v1.0 never imports them; and the retry and")
print("the breaker sit unused inside a file that is. Chapter 31's review found both.")
print()
print("Read it in this order: config, then tools, then agent, then clarity. Four")
print("files and you have the system; the rest is what it took to trust it.")

json.dump({"repository": everything, "running": running, "unused_platform": unused,
           "rows": rows}, open("code/32/_tour.json", "w"), indent=1)
