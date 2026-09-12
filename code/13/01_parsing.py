# Where real retrieval projects die. Not in the model — in the loader.

import json
from pathlib import Path

import pdfplumber
import pypdf

DOCS = Path("data/meridian/documents")
CASES = [
    ("an ordinary report", DOCS / "quarterly-reviews-pdf" / "qbr-2024-Q3.pdf"),
    ("a two-column page", DOCS / "awkward" / "01-newsletter-two-column.pdf"),
    ("a scan", DOCS / "awkward" / "06-goods-received-note-SCANNED.pdf"),
]

report = {}
for label, path in CASES:
    with open(path, "rb") as f:
        pages = pypdf.PdfReader(f).pages
        naive = "\n".join(p.extract_text() or "" for p in pages)
    with pdfplumber.open(path) as pdf:
        careful = "\n".join(p.extract_text() or "" for p in pdf.pages)
        tables = sum(len(p.extract_tables()) for p in pdf.pages)

    report[label] = {"file": path.name, "pages": len(pages),
                     "pypdf_chars": len(naive.strip()),
                     "pdfplumber_chars": len(careful.strip()), "tables": tables}
    print(f"{label:22} {path.name}")
    print(f"   pages {len(pages)}   pypdf {len(naive.strip()):>6,} chars   "
          f"pdfplumber {len(careful.strip()):>6,} chars   tables found {tables}")
    first = (careful.strip().split("\n") or [""])[:2]
    print(f"   first lines: {first}")
    print()

Path("code/13/_parsing.json").write_text(json.dumps(report, indent=2))

print("Three documents, three different problems.\n")
print("The ordinary report extracts cleanly, and the table inside it comes out as")
print("text — the columns become a run of numbers with the alignment gone. Chapter 10")
print("showed what that does once it is chunked.")
print()
print("The two-column page extracts in the wrong order. Both columns are read as one")
print("stream, so a sentence from the left column is followed by an unrelated sentence")
print("from the right. It is fluent, plausible, and not what the page says.")
print()
print("The scan extracts nothing at all. Zero characters. There is no text layer to")
print("find, because there is no text — only an image of some. This is the failure")
print("that produces an empty index and a search that returns nothing, with no error")
print("anywhere in the pipeline.")
