# Clarity v0.1, run against a real Meridian document.

import sys
from pathlib import Path

sys.path.insert(0, "code")
from clarity.v0_1.summarize import summarize          # noqa: E402

path = Path("data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md")
summary, prompt_tokens, completion_tokens = summarize(path.read_text())

print(f"{path.name}\n")
print(summary)
print(f"\n[{prompt_tokens} prompt + {completion_tokens} completion tokens]")
