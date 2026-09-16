# timeout: 600
# A prompt's regression tests: properties every output must have, checked by code, run
# against every version before it is promoted.

import re
from pathlib import Path

from openai import OpenAI

from clarity.config import MODEL_FAST
from clarity.prompts import load

client = OpenAI()
REVIEWS = ["qbr-2023-Q4.md", "qbr-2024-Q3.md", "qbr-2025-Q2.md"]


def sentences(text: str) -> int:
    return len(re.findall(r"[.!?](?:\s|$)", text.strip()))


def figures_in(text: str) -> set[str]:
    return {f.rstrip(",.") for f in re.findall(r"\$?\d[\d,]*(?:\.\d+)?%?", text)}


def checks(summary: str, source: str) -> dict[str, bool]:
    invented = {f for f in figures_in(summary) if f.strip("$%.,") not in source.replace(",", "")
                and f not in source}
    return {
        "3 sentences": sentences(summary) == 3,
        "no bullets": not re.search(r"^\s*[-*•]", summary, re.M),
        "figures sourced": not invented,
        "< 120 words": len(summary.split()) < 120,
    }


print(f"  {'version':8} {'review':16} " + " ".join(f"{name:>15}" for name in
                                                  checks("x.", "x")))
passed = {}
for version in (1, 2):
    system = load("summarize", version=version)
    for name in REVIEWS:
        source = Path("data/meridian/documents/quarterly-reviews", name).read_text()
        summary = client.chat.completions.create(
            model=MODEL_FAST, temperature=0,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": source}],
        ).choices[0].message.content or ""
        result = checks(summary, source)
        passed.setdefault(version, []).append(all(result.values()))
        marks = " ".join(f"{'pass' if ok else 'FAIL':>15}" for ok in result.values())
        print(f"  v{version:<7} {name:16} {marks}")

for version, oks in passed.items():
    print(f"\nv{version}: {sum(oks)} of {len(oks)} reviews pass every check")
