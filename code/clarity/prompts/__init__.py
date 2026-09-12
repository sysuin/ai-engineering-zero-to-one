"""
Prompts live in files, not in code.

Three reasons, and all three arrive within a month of starting:

1.  A prompt is the part of the system most likely to change, and the part least likely
    to be changed by a programmer. Keeping it in a text file means a domain expert can
    edit it without touching Python.
2.  A prompt in a file has a version history, so when quality drops you can see what
    changed. A prompt in a string literal has one too, buried in a diff of unrelated code.
3.  From Chapter 22, every prompt change is tested. A test needs something to name, and
    `summarize@v2` is a name.

    from clarity.prompts import load
    system = load("summarize", version=2)
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

HERE = Path(__file__).parent


class PromptNotFound(FileNotFoundError):
    """Raised with the list of what does exist, which is what you wanted to know."""


@lru_cache(maxsize=None)
def load(name: str, version: int | None = None) -> str:
    """
    Return the text of a prompt.

    With no version, the highest-numbered one wins — so adding `v3.md` promotes it
    everywhere, and deleting it rolls the whole system back.
    """
    candidates = sorted(HERE.glob(f"{name}/v*.md"),
                        key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    if not candidates:
        available = sorted(p.name for p in HERE.iterdir()
                           if p.is_dir() and not p.name.startswith(("_", ".")))
        raise PromptNotFound(
            f"No prompt named {name!r}. Available: {', '.join(available) or 'none'}")

    if version is None:
        chosen = candidates[-1]
    else:
        chosen = next((p for p in candidates if p.stem == f"v{version}"), None)
        if chosen is None:
            have = ", ".join(p.stem for p in candidates)
            raise PromptNotFound(f"{name!r} has no {f'v{version}'!r}. Have: {have}")

    return chosen.read_text().strip()


def versions(name: str) -> list[int]:
    """Which versions of a prompt exist, oldest first."""
    return sorted(int(re.findall(r"\d+", p.stem)[0])
                  for p in HERE.glob(f"{name}/v*.md"))
