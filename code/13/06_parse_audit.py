# A parse audit: run the extractor over every PDF, and flag the documents whose text does not
# look like text. Cheap, local, and the check most pipelines never run.

import re
import statistics
from pathlib import Path

from pypdf import PdfReader

DOCS = Path("data/meridian/documents")
pdfs = sorted(DOCS.glob("*/*.pdf"))


def audit(path: Path) -> dict:
    reader = PdfReader(path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    lines = [l for l in text.splitlines() if l.strip()]
    words = re.findall(r"[A-Za-z]{2,}", text)
    flags = []
    if len(text.strip()) < 100 * len(reader.pages):
        flags.append("almost no text: a scan?")
    source = path.parent.parent / path.parent.name.replace("-pdf", "") / f"{path.stem}.md"
    if not source.exists():
        source = path.with_suffix(".md")
    if source.exists() and text.strip():
        source_words = re.findall(r"[A-Za-z]{2,}", source.read_text())
        missing = 1 - len(set(source_words) & set(words)) / max(1, len(set(source_words)))
        if missing > 0.15:
            flags.append(f"{missing:.0%} of the source's words absent")
        # Order, not just presence: how many adjacent word pairs of the source survive.
        pairs = set(zip(source_words, source_words[1:]))
        kept = len(pairs & set(zip(words, words[1:]))) / max(1, len(pairs))
        if kept < 0.85:
            flags.append(f"only {kept:.0%} of word pairs in order: columns or a table?")
    return {"pages": len(reader.pages), "chars": len(text),
            "median_line": statistics.median(len(l) for l in lines) if lines else 0,
            "flags": flags}


flagged = 0
print(f"{len(pdfs)} PDFs\n")
print(f"  {'document':40} {'pages':>5} {'chars':>7}  flags")
for path in pdfs:
    r = audit(path)
    if r["flags"] or path.parent.name == "awkward":
        print(f"  {path.name[:40]:40} {r['pages']:>5} {r['chars']:>7,}  "
              + "; ".join(r["flags"]))
    flagged += bool(r["flags"])
print(f"\n{flagged} of {len(pdfs)} flagged; the rest are not listed")
