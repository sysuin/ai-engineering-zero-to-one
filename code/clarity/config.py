"""
Every setting in one place — and the only place in this book that names a model.

Two rules govern this file:

1.  **No model name or price appears in any chapter.** Chapters refer to `MODEL_FAST`,
    `MODEL_SMART` and `MODEL_EMBED`. When a provider ships something better, you change
    three lines here and the whole book is current again. Appendix C lists what these
    resolve to today and what each one costs.

2.  **No secret appears in any file that git can see.** Keys are read from the environment,
    which is loaded from a `.env` file that `.gitignore` excludes.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# The project root: two levels up from this file (code/clarity/config.py).
ROOT = Path(__file__).resolve().parents[2]

# Load .env if it is there. Real environment variables always win, so a value exported
# in your shell — or set by your CI — overrides the file.
load_dotenv(ROOT / ".env", override=False)


# --------------------------------------------------------------------------- models
# Change these three lines, not the chapters.

MODEL_FAST = os.getenv("MODEL_FAST", "gpt-5.4-mini")
"""Cheap and quick. The default. Most of this book runs on it."""

MODEL_SMART = os.getenv("MODEL_SMART", "gpt-5.5")
"""
Slower, dearer, and better at reasoning. Used only where a chapter shows why it is
worth it.

Note that it does not accept every parameter `MODEL_FAST` does — see CAPABILITIES.
That is not a defect: a model that decides how long to think about something has no
use for a knob that tells it how random to be.
"""

MODEL_EMBED = os.getenv("MODEL_EMBED", "text-embedding-3-small")
"""Turns text into vectors. Part III."""


# --------------------------------------------------------------------------- capabilities
# Not every model accepts every parameter, and the differences are not cosmetic.
# Chapters check this rather than assuming. Appendix C carries the full table.
CAPABILITIES: dict[str, set[str]] = {
    MODEL_FAST:  {"temperature", "top_p", "seed", "logprobs", "max_completion_tokens"},
    MODEL_SMART: {"seed", "max_completion_tokens", "reasoning_effort"},
    MODEL_EMBED: {"dimensions"},
}


def supports(model: str, parameter: str) -> bool:
    """Whether `model` accepts `parameter`. Unknown models are assumed capable."""
    return parameter in CAPABILITIES.get(model, {parameter})


# --------------------------------------------------------------------------- secrets
class MissingKey(RuntimeError):
    """Raised with instructions rather than a stack trace nobody can act on."""


def openai_key() -> str:
    """
    Return the OpenAI API key, or fail with something a person can act on.

    Failing loudly here is deliberate. A missing key that surfaces as a 401 six function
    calls deeper is a bad afternoon.
    """
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key or key.startswith("sk-your-key"):
        raise MissingKey(
            "OPENAI_API_KEY is not set.\n\n"
            f"  1. Open {ROOT / '.env'}\n"
            "  2. Replace the placeholder on the OPENAI_API_KEY line with your key\n"
            "  3. Save, and run this again\n\n"
            "The file is listed in .gitignore, so the key cannot be committed.\n"
            "If you do not have a key yet, Appendix B walks through getting one — and\n"
            "setting a spending limit before you make your first call."
        )
    return key


def fingerprint(key: str) -> str:
    """A safe way to refer to a key in a log, a ticket, or a book."""
    return f"{key[:3]}…{key[-4:]} ({len(key)} chars)"


# --------------------------------------------------------------------------- spending
SPEND_CEILING_USD = float(os.getenv("BOOK_SPEND_CEILING_USD", "2.00"))
"""A rail for the build: one full pass over every listing must cost less than this."""


def rate(model: str) -> tuple[float, float] | None:
    """
    Dollars per million tokens, (input, output), read from the environment.

    Deliberately not hard-coded. Prices change, and a wrong number printed with
    confidence is worse than no number at all. Set them in `.env` from the provider's
    current pricing page; Appendix C says where to look. When they are not set, the
    book reports tokens and stays silent about dollars.
    """
    key = model.upper().replace("-", "_").replace(".", "_")
    raw_in = os.getenv(f"RATE_{key}_INPUT")
    raw_out = os.getenv(f"RATE_{key}_OUTPUT")
    if raw_in and raw_out:
        return float(raw_in), float(raw_out)
    return None
