#!/usr/bin/env python3
# skip
"""
Everything this book needs, checked in the order it will break.

Run it before Chapter 4. Each check prints a line and, if it fails, what to do about it —
and the script stops at the first failure, because every check below a failure would only
produce noise about a cause you already know.

    python3 code/00_check_setup.py
"""
from __future__ import annotations

import importlib
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

OK, BAD, WARN = "  ok  ", " FAIL ", " note "
failures: list[str] = []


class Advisory(RuntimeError):
    """A problem worth saying out loud that should not stop the run."""


def check(label: str, fn) -> bool:
    """Run one check. Returns True to continue, False to stop the run."""
    try:
        detail = fn()
    except Advisory as note:
        print(f"[{WARN}] {label}")
        print(f"         {note}")
        return True
    except Exception as error:                                   # noqa: BLE001
        print(f"[{BAD}] {label}")
        print(f"         {type(error).__name__}: {error}")
        failures.append(label)
        return False
    print(f"[{OK}] {label:<34}{detail}")
    return True


# ------------------------------------------------------------------ 1. the language
def python_version() -> str:
    if sys.version_info < (3, 10):
        raise RuntimeError(
            f"Python {platform.python_version()} is too old; this book needs 3.10+.\n"
            "         Appendix B has the install for your platform."
        )
    return f"{platform.python_version()} on {platform.system()}"


# ------------------------------------------------------------------ 2. the venv
def virtualenv() -> str:
    active = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    where = Path(sys.executable)
    if not active:
        # A note rather than a failure: conda and a system Python with the packages
        # installed both work fine. It is still the first thing to check when an
        # import fails, which is why it is here at all.
        raise Advisory(
            f"No virtual environment is active. Python is {where}.\n"
            "         That is fine if you meant it. If a later import fails, this is\n"
            "         the first thing to fix:\n"
            "           python3 -m venv .venv && source .venv/bin/activate\n"
            "           (Windows: .venv\\Scripts\\activate)"
        )
    return str(where)


# ------------------------------------------------------------------ 3. the packages
REQUIRED = {
    "openai": "openai", "pydantic": "pydantic", "dotenv": "python-dotenv",
    "numpy": "numpy", "yaml": "pyyaml", "tiktoken": "tiktoken",
}


def packages() -> str:
    missing = [dist for module, dist in REQUIRED.items()
               if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError(
            f"Not installed: {', '.join(missing)}\n"
            "         Run:  python3 -m pip install -r requirements.txt\n"
            "         Use `python3 -m pip`, never bare `pip` — they can point at\n"
            "         different environments, which is its own afternoon."
        )
    import openai
    return f"{len(REQUIRED)} present, openai {openai.__version__}"


# ------------------------------------------------------------------ 4. the key
def api_key() -> str:
    from clarity.config import fingerprint, openai_key
    return fingerprint(openai_key())


# ------------------------------------------------------------------ 5. it authenticates
def authenticates() -> str:
    from openai import OpenAI

    from clarity.config import MODEL_FAST
    response = OpenAI().chat.completions.create(
        model=MODEL_FAST,
        messages=[{"role": "user", "content": "Reply with the single word: ready"}],
        max_completion_tokens=16,
    )
    usage = response.usage
    return (f"{MODEL_FAST} answered, "
            f"{usage.prompt_tokens}+{usage.completion_tokens} tokens")


# ------------------------------------------------------------------ 6. the dataset
def dataset() -> str:
    docs = ROOT / "data" / "meridian" / "documents"
    warehouse = ROOT / "data" / "meridian" / "warehouse" / "meridian.db"
    if not docs.exists() or not warehouse.exists():
        raise RuntimeError(
            "The Meridian dataset is not built.\n"
            "         Run:  python3 code/meridian/generate.py\n"
            "         Then: python3 code/meridian/verify.py"
        )
    n = sum(1 for _ in docs.rglob("*.md"))
    return f"{n} documents, warehouse {warehouse.stat().st_size / 1e6:.1f} MB"


# ------------------------------------------------------------------ 7. the index
def index() -> str:
    cache = ROOT / "data" / "meridian" / "index" / "vectors.npy"
    if not cache.exists():
        raise RuntimeError(
            "No vector index yet. This is the one setup step that costs money — a\n"
            "         few cents — and it is not needed until Chapter 11.\n"
            "         Run:  python3 code/meridian_index.py --build"
        )
    import numpy as np
    vectors = np.load(cache, mmap_mode="r")
    return f"{vectors.shape[0]:,} chunks x {vectors.shape[1]} dimensions"


CHECKS = [
    ("Python 3.10 or newer", python_version),
    ("a virtual environment is active", virtualenv),
    ("the required packages import", packages),
    ("an API key is loaded from .env", api_key),
    ("the key authenticates", authenticates),
    ("the Meridian dataset exists", dataset),
    ("the vector index is built", index),
]

print("Checking your setup. Each line is a thing a later chapter depends on.\n")
for label, fn in CHECKS:
    if not check(label, fn):
        break

print()
if failures:
    print(f"Stopped at: {failures[0]}")
    print("Fix that one and run this again. Appendix B has the long version.")
    raise SystemExit(1)
print("Everything checks out. Chapter 4 will work.")
