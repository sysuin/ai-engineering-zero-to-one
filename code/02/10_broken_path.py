# expect-fail
# Failure 4 of 4: a file that is not where you said it was. Note which quarter is asked for.

from pathlib import Path

review = Path("data/meridian/documents/quarterly-reviews/qbr-2026-Q1.md")
print(review.read_text()[:200])
