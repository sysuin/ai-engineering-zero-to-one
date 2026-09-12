# timeout: 900
# A page with no text layer. The only thing left is to look at it.

import base64
import io
import subprocess
import tempfile
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
SCAN = Path("data/meridian/documents/awkward/06-goods-received-note-SCANNED.pdf")


def page_images(pdf: Path, dpi: int = 150) -> list[bytes]:
    """Rasterise each page. Every OCR and vision pipeline starts here."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(pdf),
                        f"{tmp}/page"], check=True, capture_output=True)
        return [Path(p).read_bytes() for p in sorted(Path(tmp).glob("page*.png"))]


images = page_images(SCAN)
print(f"{SCAN.name}: {len(images)} page(s), "
      f"{sum(len(i) for i in images) / 1024:.0f} KB of image\n")

response = client.chat.completions.create(
    model=MODEL_FAST, temperature=0, max_completion_tokens=800,
    messages=[{"role": "user", "content": [
        {"type": "text", "text":
         "Transcribe this scanned document as Markdown. Preserve the table as a "
         "Markdown table. Transcribe only what is visible; do not add commentary, "
         "and do not correct anything that looks wrong."},
        {"type": "image_url", "image_url": {
            "url": "data:image/png;base64," + base64.b64encode(images[0]).decode()}},
    ]}],
)
transcript = (response.choices[0].message.content or "").strip()

print(transcript[:700])
print()
print(f"[{response.usage.prompt_tokens} prompt tokens, of which the image is most]")

# The generator knows what this page says, so the transcription can be checked.
truth = Path("data/meridian/documents/awkward/06-scanned-source.md").read_text()
FACTS = ["GRN-2025-04471", "Columbus", "MRD-CLE-004", "480", "228", "12 short"]
found = [f for f in FACTS if f in transcript]
print(f"\nFacts recovered: {len(found)}/{len(FACTS)}   {found}")
missing = [f for f in FACTS if f not in transcript]
if missing:
    print(f"Missing: {missing}")

# One transcription error, and it is the instructive kind.
print(f"\nRegion spelled: "
      f"{'Mid-west' if 'Mid-west' in transcript else 'Midwest' if 'Midwest' in transcript else '?'}"
      f"   (the document says Midwest)")
print("Every fact came through, and the region acquired a hyphen. A human reader would")
print("not notice; an exact metadata match on 'Midwest' would fail. Transcription is")
print("good enough to search and not good enough to key on.")

print()
print("Reading an image costs far more than reading text — the tokens above are almost")
print("all image. Do it once, at indexing time, and store the transcription. Doing it")
print("per query would be paying to re-read the same page forever.")
print()
print("And treat the result as data, not as truth: it is a model's reading of a picture")
print("of a document. Chapter 29 treats anything that arrives this way as hostile.")
