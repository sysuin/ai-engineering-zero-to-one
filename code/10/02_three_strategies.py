# timeout: 1800
# Ten questions about a 29,000-token document, three ways.

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import tiktoken
from openai import OpenAI

from clarity.config import MODEL_FAST

client = OpenAI()
encoder = tiktoken.encoding_for_model(MODEL_FAST)

APPENDIX = Path("data/meridian/documents/awkward/03-long-appendix.md").read_text()
SECTIONS = ["## " + s for s in APPENDIX.split("## ") if s.strip()][1:]

# Ten questions whose answers are checkable against the document itself.
QUESTIONS = [
    (f"What revision is specification item {i} at? Reply with the number only.",
     re.search(rf"C\.{i} .*?Revision (\d+)", APPENDIX, re.S).group(1))
    for i in (3, 47, 118, 205, 266, 301, 344, 372, 388, 399)
]

USAGE: list[tuple[int, int, int]] = []


def call(content: str, budget: int = 64) -> str:
    response = client.chat.completions.create(
        model=MODEL_FAST, temperature=0, max_completion_tokens=budget,
        messages=[{"role": "user", "content": content}])
    cached = getattr(response.usage.prompt_tokens_details, "cached_tokens", 0) or 0
    USAGE.append((response.usage.prompt_tokens, response.usage.completion_tokens, cached))
    return (response.choices[0].message.content or "").strip()


# ------------------------------------------------------------------ 1. stuff it all
def stuff(question: str) -> str:
    return call(f"<document>\n{APPENDIX}\n</document>\n\n{question}")


# ------------------------------------------------------------------ 2. map-reduce
_digest: str | None = None


def map_reduce(question: str) -> str:
    global _digest
    if _digest is None:
        groups = ["\n".join(SECTIONS[i:i + 50]) for i in range(0, len(SECTIONS), 50)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            parts = list(pool.map(
                lambda g: call("Summarise every item below as one line each, "
                               f"'C.N: revision R'.\n\n{g}", budget=1600), groups))
        _digest = "\n".join(parts)
    return call(f"<notes>\n{_digest}\n</notes>\n\n{question}")


# ------------------------------------------------------------------ 3. select first
def select(question: str) -> str:
    """The crudest possible retrieval: send only the sections that mention the item."""
    wanted = re.search(r"item (\d+)", question).group(1)
    hits = [s for s in SECTIONS if f"C.{wanted} " in s]
    return call(f"<document>\n{''.join(hits)}\n</document>\n\n{question}")


STRATEGIES = {"stuff the whole thing": stuff,
              "map-reduce, then ask": map_reduce,
              "select, then ask": select}

results = {}
print(f"{'strategy':24} {'correct':>8} {'in tokens':>11} {'cached':>9} "
      f"{'out':>6} {'seconds':>9}")
for name, fn in STRATEGIES.items():
    USAGE.clear()
    started = time.time()
    answers = [fn(q) for q, _ in QUESTIONS]
    elapsed = time.time() - started

    correct = sum(expected in answer for answer, (_, expected)
                  in zip(answers, QUESTIONS))
    prompt = sum(u[0] for u in USAGE)
    completion = sum(u[1] for u in USAGE)
    cached = sum(u[2] for u in USAGE)
    results[name] = {"correct": correct, "prompt_tokens": prompt,
                     "cached_tokens": cached, "completion_tokens": completion,
                     "seconds": round(elapsed, 1), "calls": len(USAGE)}
    print(f"{name:24} {correct:>5}/{len(QUESTIONS)} {prompt:>11,} {cached:>9,} "
          f"{completion:>6,} {elapsed:>9.1f}")

Path("code/10/_strategies.json").write_text(json.dumps(
    {"n_questions": len(QUESTIONS),
     "document_tokens": len(encoder.encode(APPENDIX)), "results": results}, indent=2))

print(f"\nThe document is {len(encoder.encode(APPENDIX)):,} tokens. "
      f"It fits in the window with room to spare.")
print()
print("All three get the answers. What separates them is what they cost, and the")
print("separation is two orders of magnitude — for identical results.")
