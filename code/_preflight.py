#!/usr/bin/env python3
"""
Check that everything the book needs is in place, before you spend an afternoon
discovering it is not.

    python3 code/_preflight.py

Never prints your key. The most it will show you is a fingerprint: the first three
characters, the last four, and the length — enough to tell two keys apart, and useless
to anyone who reads it.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))

CHECKS: list[tuple[str, bool, str]] = []


def check(label: str, ok: bool, detail: str = "") -> bool:
    CHECKS.append((label, ok, detail))
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  — {detail}" if detail else ""))
    return ok


def main() -> int:
    print("\nEnvironment")
    check("Python 3.11 or newer", sys.version_info >= (3, 11),
          ".".join(map(str, sys.version_info[:3])))

    for package in ("requests", "dotenv", "pandas"):
        try:
            __import__(package)
            check(f"{package} installed", True)
        except ImportError:
            check(f"{package} installed", False, "pip install -r requirements.txt")

    print("\nDataset")
    from clarity.config import ROOT
    db = ROOT / "data" / "meridian" / "warehouse" / "meridian.db"
    check("Meridian warehouse present", db.exists(),
          str(db.relative_to(ROOT)) if db.exists()
          else "run: python3 code/meridian/generate.py")

    print("\nSecrets")
    env_file = ROOT / ".env"
    check(".env exists", env_file.exists(), str(env_file))

    import subprocess
    ignored = subprocess.run(["git", "check-ignore", ".env"], cwd=ROOT,
                             capture_output=True).returncode == 0
    check(".env is ignored by git", ignored,
          "" if ignored else "add '.env' to .gitignore before doing anything else")

    from clarity.config import MissingKey, fingerprint, openai_key
    try:
        key = openai_key()
        check("OPENAI_API_KEY is set", True, fingerprint(key))
    except MissingKey:
        check("OPENAI_API_KEY is set", False, "see .env — the key is still the placeholder")
        key = None

    if key:
        print("\nProvider")
        try:
            from openai import OpenAI
        except ImportError:
            check("openai package installed", False, "pip install openai")
            return 1
        check("openai package installed", True)
        try:
            client = OpenAI(api_key=key)
            models = client.models.list()
            check("key works", True, f"{len(models.data)} models available")
        except Exception as error:                       # noqa: BLE001
            name = type(error).__name__
            hint = {"AuthenticationError": "the key is wrong or has been revoked",
                    "PermissionDeniedError": "the key is valid but lacks access",
                    "RateLimitError": "valid key, but you are out of quota or credit"}
            check("key works", False, f"{name}: {hint.get(name, str(error)[:70])}")

    failed = [label for label, ok, _ in CHECKS if not ok]
    print(f"\n  {len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print(f"  Fix first: {failed[0]}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
