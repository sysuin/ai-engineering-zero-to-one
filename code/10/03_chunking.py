# Five ways of cutting a document up, and what each one breaks.

import json
import re
from pathlib import Path

import tiktoken

from clarity.config import MODEL_FAST

encoder = tiktoken.encoding_for_model(MODEL_FAST)
AWKWARD = Path("data/meridian/documents/awkward")
NOISY = (AWKWARD / "04-header-footer-noise.md").read_text()
TABLE = (AWKWARD / "02-rotated-table.md").read_text()
APPENDIX = (AWKWARD / "03-long-appendix.md").read_text()

TARGET = 200          # tokens per chunk, roughly


def fixed_chars(text: str, size: int = 800) -> list[str]:
    return [text[i:i + size] for i in range(0, len(text), size)]


def fixed_tokens(text: str, size: int = TARGET) -> list[str]:
    ids = encoder.encode(text)
    return [encoder.decode(ids[i:i + size]) for i in range(0, len(ids), size)]


def by_paragraph(text: str, size: int = TARGET) -> list[str]:
    chunks, current = [], ""
    for para in text.split("\n\n"):
        if len(encoder.encode(current + para)) > size and current:
            chunks.append(current.strip())
            current = ""
        current += para + "\n\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


def by_structure(text: str, size: int = TARGET) -> list[str]:
    """Split on headings first, and only fall back to paragraphs inside a long one."""
    parts = re.split(r"(?=^#{1,3} )", text, flags=re.M)
    chunks = []
    for part in parts:
        if not part.strip():
            continue
        if len(encoder.encode(part)) <= size:
            chunks.append(part.strip())
        else:
            chunks.extend(by_paragraph(part, size))
    return chunks


def with_overlap(text: str, size: int = TARGET, overlap: int = 40) -> list[str]:
    ids = encoder.encode(text)
    step = size - overlap
    return [encoder.decode(ids[i:i + size]) for i in range(0, len(ids), step)]


STRATEGIES = {"fixed characters": fixed_chars, "fixed tokens": fixed_tokens,
              "by paragraph": by_paragraph, "by structure": by_structure,
              "fixed + overlap": with_overlap}

print("Cutting the 400-item appendix five ways\n")
print(f"{'strategy':20} {'chunks':>7} {'median':>8} {'split a section':>17}")
report = {}
for name, fn in STRATEGIES.items():
    chunks = fn(APPENDIX)
    sizes = sorted(len(encoder.encode(c)) for c in chunks)
    # A chunk that starts mid-item has lost its heading, and with it the item number.
    orphaned = sum(1 for c in chunks if not c.lstrip().startswith(("#", "C.")))
    report[name] = {"chunks": len(chunks), "median_tokens": sizes[len(sizes) // 2],
                    "orphaned": orphaned}
    print(f"{name:20} {len(chunks):>7} {sizes[len(sizes) // 2]:>8} "
          f"{orphaned:>13} ({orphaned / len(chunks):.0%})")

print("\nAn orphaned chunk begins mid-item, so it no longer says which item it is about.")
print("Retrieval can still find it. It just cannot tell you what it refers to.\n")

# The noisy document: the same banner on all 24 pages.
print("The document with a banner on every page")
chunks = fixed_tokens(NOISY, 120)
banner = "MERIDIAN SUPPLY CO. — CONFIDENTIAL"
polluted = sum(banner in c for c in chunks)
print(f"  {len(chunks)} chunks, {polluted} contain the same confidentiality banner")
print(f"  the banner is {len(encoder.encode(banner))} tokens, repeated "
      f"{NOISY.count(banner)} times in the source")
print("  every chunk embedded from this document carries it, so they all look alike")

# The table: what fixed-size chunking does to a table.
print("\nThe table, cut at 120 tokens")
for i, chunk in enumerate(fixed_tokens(TABLE, 120)):
    first = chunk.strip().split("\n")[0][:62]
    print(f"  chunk {i}: {first!r}")
print("  the second chunk is rows without a header. The numbers survive; the meaning")
print("  does not.")

Path("code/10/_chunking.json").write_text(json.dumps(report, indent=2))
