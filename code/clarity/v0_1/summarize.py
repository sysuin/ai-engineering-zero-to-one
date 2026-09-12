"""
Clarity v0.1 — summarise one Meridian document from the command line.

    python3 code/clarity/v0_1/summarize.py data/meridian/documents/quarterly-reviews/qbr-2024-Q3.md

Everything this book builds starts here. It is deliberately small, and it is deliberately
honest about what it costs: the last line reports the tokens, because a system that cannot
tell you what it spent is a system you cannot run.
"""
from __future__ import annotations

import sys
from pathlib import Path

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from clarity.config import MODEL_FAST, rate      # noqa: E402

SYSTEM = (
    "You summarise internal business documents for an analytics team. "
    "Be specific and factual. Use only what the document says. "
    "If a figure is not in the document, do not state it."
)


def summarize(text: str, sentences: int = 3) -> tuple[str, int, int]:
    """Return a summary of `text`, with the prompt and completion token counts."""
    client = OpenAI()
    response = client.chat.completions.create(
        model=MODEL_FAST,
        messages=[
            {"role": "system", "content": SYSTEM},
            {"role": "user",
             "content": f"Summarise the following document in {sentences} sentences.\n\n"
                        f"---\n{text}\n---"},
        ],
    )
    usage = response.usage
    return (response.choices[0].message.content.strip(),
            usage.prompt_tokens, usage.completion_tokens)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2

    path = Path(argv[1])
    if not path.exists():
        print(f"No such file: {path}")
        return 1

    summary, prompt_tokens, completion_tokens = summarize(path.read_text())

    print(f"{path.name}\n")
    print(summary)

    rates = rate(MODEL_FAST)
    cost = ""
    if rates:
        cost = f", ${(prompt_tokens * rates[0] + completion_tokens * rates[1]) / 1e6:.6f}"
    print(f"\n[{prompt_tokens} prompt + {completion_tokens} completion tokens{cost}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
