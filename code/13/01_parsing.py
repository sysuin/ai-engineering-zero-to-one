# Where real retrieval projects die. Not in the model — in the loader.

import json
import re
from pathlib import Path

import pdfplumber
import pypdf

DOCS = Path("data/meridian/documents")
CASES = [
    ("an ordinary report", DOCS / "quarterly-reviews-pdf" / "qbr-2024-Q3.pdf",
     DOCS / "quarterly-reviews" / "qbr-2024-Q3.md"),
    ("a two-column page", DOCS / "awkward" / "01-newsletter-two-column.pdf",
     DOCS / "awkward" / "01-newsletter-two-column.md"),
    ("a scan", DOCS / "awkward" / "06-goods-received-note-SCANNED.pdf",
     DOCS / "awkward" / "06-scanned-source.md"),
]


def problems(text: str, source: str) -> list[str]:
    """What is wrong with an extraction, found by comparing it with the text it came from."""
    found = []
    if not text.strip():
        return ["no text at all"]
    words = re.findall(r"[A-Za-z]+", text)
    source_words = re.findall(r"[A-Za-z]+", source)
    pairs = set(zip(source_words, source_words[1:]))
    in_order = len(pairs & set(zip(words, words[1:]))) / max(1, len(pairs))
    found.append(f"{in_order:.0%} of the source's word pairs survive in order")
    merged = [w for w in re.findall(r"\S+", text) if re.fullmatch(r"[A-Za-z$,.%0-9]{22,}", w)]
    if merged:
        found.append(f"{len(merged)} run(s) of words with the spaces lost, e.g. {merged[0][:24]!r}")
    ligatures = len(re.findall("[\ufb00-\ufb06]", text))
    if ligatures:
        found.append(f"{ligatures} ligature character(s): 'office' "
                     "arrives as 'o' + U+FB00 + 'ice'")
    hyphens = len(re.findall(r"[a-z]-\n[a-z]", text))
    if hyphens:
        found.append(f"{hyphens} word(s) broken across lines by a hyphen")
    return found


report = {}
for label, path, source_path in CASES:
    source = source_path.read_text()
    with open(path, "rb") as f:
        naive = "\n".join(p.extract_text() or "" for p in pypdf.PdfReader(f).pages)
    with pdfplumber.open(path) as pdf:
        careful = "\n".join(p.extract_text() or "" for p in pdf.pages)
        tables = sum(len(p.extract_tables()) for p in pdf.pages)
    report[label] = {"file": path.name, "pypdf_chars": len(naive.strip()),
                     "pdfplumber_chars": len(careful.strip()), "tables": tables,
                     "pypdf": problems(naive, source), "pdfplumber": problems(careful, source)}
    print(f"{label} — {path.name}")
    print(f"   tables recognised by pdfplumber: {tables}")
    for tool, text in (("pypdf", naive), ("pdfplumber", careful)):
        print(f"   {tool:10} {len(text.strip()):>6,} chars: " + "; ".join(report[label][tool]))
    print()

Path("code/13/_parsing.json").write_text(json.dumps(report, indent=2))

print("Nothing above raised an exception. Every problem was found only by comparing the")
print("extracted text with what the document actually says.")
