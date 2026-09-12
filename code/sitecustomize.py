"""
Token accounting for the book's build, with no trace in the teaching code.

Python imports `sitecustomize` automatically at interpreter startup when it is on the
path. `code/_runner.py` puts `code/` on PYTHONPATH; nothing else does. So when the build
runs a listing, every model call is counted — and when a *reader* runs the same listing,
this file is never imported and the listing behaves exactly as printed in the book.

That property is the point. Instrumenting the listings themselves would mean the code in
the book is not the code that runs.
"""
from __future__ import annotations

import atexit
import json
import os
import time

if os.getenv("BOOK_RUN") == "1":
    _records: list[dict] = []

    def _install() -> None:
        try:
            from openai.resources.chat import completions
        except Exception:                                   # noqa: BLE001
            return

        original = completions.Completions.create

        def counted(self, *args, **kwargs):
            started = time.time()
            response = original(self, *args, **kwargs)
            usage = getattr(response, "usage", None)
            if usage is not None:
                _records.append({
                    "model": kwargs.get("model", "?"),
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                    "seconds": round(time.time() - started, 3),
                })
            return response

        completions.Completions.create = counted

    def _flush() -> None:
        if not _records:
            return
        path = os.getenv("BOOK_USAGE_FILE", "code/_usage.jsonl")
        listing = os.getenv("BOOK_LISTING", "unknown")
        with open(path, "a") as f:
            for record in _records:
                f.write(json.dumps({"listing": listing, **record}) + "\n")

    _install()
    atexit.register(_flush)
